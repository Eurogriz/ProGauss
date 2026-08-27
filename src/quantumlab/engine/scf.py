"""Ограниченный по спину метод Хартри–Фока (RHF) с ускорением DIIS.

Референсная реализация: плотные матрицы, каноническая ортогонализация
``S^{-1/2}``, прямое построение фока. Задача модуля — дать **проверяемый**
эталон (ADR-002), поэтому порядок стратегий ускорения фиксирован и
протоколируется, а не подбирается эвристически.

Порядок стратегий (§10):
  1. стартовая плотность из диагонализации одноэлектронного гамильтониана;
  2. DIIS (Pulay) с 3-й итерации;
  3. если DIIS расходится или осциллирует — гашение (damping);
  4. если и это не помогает — сдвиг уровней (level shift);
  5. если сходимость не достигнута — честный результат ``converged = False``.

Что сознательно не реализовано в этом срезе: EDIIS, проверка устойчивости
волновой функции, дробные занятия. Они объявлены в реестре как
``not_implemented``, а не «почти работающие».
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import BasisSet, nuclear_repulsion
from quantumlab.engine.integrals import (
    build_core_hamiltonian,
    build_electron_repulsion,
    build_overlap,
)


@dataclass(frozen=True, slots=True)
class ScfSettings:
    """Параметры итерационного процесса."""

    max_iterations: int = 128
    energy_tolerance: float = 1e-9
    density_tolerance: float = 1e-7
    diis_start: int = 2
    diis_space: int = 8
    damping_factor: float = 0.5
    damping_rounds: int = 0
    level_shift: float = 0.25
    #: Отношение, во сколько раз должна упасть невязка DIIS за ``diis_space``
    #: итераций, чтобы стратегия считалась работающей. Если нет — включается
    #: сдвиг уровней.
    stall_ratio: float = 0.5


@dataclass(slots=True)
class ScfHistory:
    """Одна итерация SCF — для протокола расчёта и отладки."""

    iteration: int
    energy: float
    energy_change: float
    density_change: float
    diis_error: float
    strategy: str


@dataclass(slots=True)
class ScfResult:
    """Результат RHF."""

    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    orbital_energies: tuple[float, ...]
    coefficients: np.ndarray
    density: np.ndarray
    converged: bool
    iterations: int
    history: list[ScfHistory] = field(default_factory=list)
    strategies_used: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0


def canonical_orthogonalizer(overlap: np.ndarray) -> np.ndarray:
    """Матрица ``X = S^{-1/2}`` через собственное разложение.

    Собственное разложение устойчивее обращения в степени -1/2 через
    разложение Холесского: оно же позволяет отбросить почти вырожденные
    направления, если базис окажется линейно зависимым.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    positive = eigenvalues > 1e-10
    inverse_sqrt = np.zeros_like(eigenvalues)
    inverse_sqrt[positive] = 1.0 / np.sqrt(eigenvalues[positive])
    return np.asarray(eigenvectors @ np.diag(inverse_sqrt) @ eigenvectors.T)


def build_fock(core: np.ndarray, density: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Фок-матрица ``F = H + J − ½K``.

    ``J_μν = Σ_λσ D_λσ (μν|λσ)``, ``K_μν = Σ_λσ D_λσ (μλ|νσ)``.
    Коэффициент ½ у обменного члена — следствие RHF (двойное занятие).
    """
    coulomb = np.einsum("ls,uvls->uv", density, eri, optimize=True)
    exchange = np.einsum("ls,ulvs->uv", density, eri, optimize=True)
    return np.asarray(core + coulomb - 0.5 * exchange)


def solve_roothaan(fock: np.ndarray, orthogonalizer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Решает уравнение Рутана в ортогонализованном базисе.

    Возвращает собственные значения и коэффициенты в исходном (неортогональном)
    базисе: ``C = X C'``.
    """
    transformed = orthogonalizer.T @ fock @ orthogonalizer
    transformed = 0.5 * (transformed + transformed.T)
    energies, coefficients_prime = np.linalg.eigh(transformed)
    return energies, orthogonalizer @ coefficients_prime


def density_from_coefficients(coefficients: np.ndarray, n_occupied: int) -> np.ndarray:
    """Плотность ``D = 2 C_occ C_occ^T`` (двойное занятие каждой МО)."""
    occupied = coefficients[:, :n_occupied]
    return np.asarray(2.0 * occupied @ occupied.T)


def electronic_energy(density: np.ndarray, core: np.ndarray, fock: np.ndarray) -> float:
    """Электронная энергия ``E = ½ Σ D(H + F)`` — без ядерного отталкивания."""
    return float(0.5 * np.sum(density * (core + fock)))


def _diis_extrapolate(
    fock_history: list[np.ndarray], error_history: list[np.ndarray]
) -> np.ndarray | None:
    """Экстраполяция Пулея. Возвращает ``None``, если система вырождена."""
    size = len(error_history)
    if size < 2:
        return None
    matrix = np.zeros((size + 1, size + 1))
    for i in range(size):
        for j in range(size):
            matrix[i, j] = float(np.sum(error_history[i] * error_history[j]))
    matrix[size, :size] = -1.0
    matrix[:size, size] = -1.0
    right = np.zeros(size + 1)
    right[size] = -1.0
    try:
        solution = np.linalg.solve(matrix, right)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(solution)):
        return None
    weights = solution[:size]
    return sum(float(w) * f for w, f in zip(weights, fock_history, strict=True))  # type: ignore[return-value]


def run_rhf(
    basis: BasisSet,
    molecule: Molecule,
    settings: ScfSettings | None = None,
) -> ScfResult:
    """Выполняет RHF-расчёт.

    Число занятых орбиталей берётся из числа электронов молекулы; расчёт
    предполагает замкнутую оболочку. Нечётное число электронов — ошибка
    вызывающей стороны (для неё нужны ROHF/UHF, которых в этом срезе нет).

    На каждой итерации выполняется ровно одна диагонализация: фокиан строится
    по текущей плотности, при необходимости экстраполируется (DIIS) и/или
    сдвигается по уровням, затем диагонализуется в ортогональном базисе.
    """
    config = settings or ScfSettings()
    started = time.perf_counter()
    strategies: list[str] = ["core-hamiltonian-guess"]

    electrons = molecule.n_electrons
    if electrons % 2 != 0:
        msg = (
            f"RHF требует чётного числа электронов (замкнутая оболочка), получено {electrons}. "
            "Для открытой оболочки нужны ROHF или UHF — они в этом срезе не реализованы."
        )
        raise ValueError(msg)
    n_occupied = electrons // 2

    overlap = build_overlap(basis, molecule)
    core = build_core_hamiltonian(basis, molecule)
    eri = build_electron_repulsion(basis, molecule)
    v_nuc = nuclear_repulsion(molecule)
    orthogonalizer = canonical_orthogonalizer(overlap)

    def diagonalize(fock_prime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symmetric = 0.5 * (fock_prime + fock_prime.T)
        energies, coefficients_prime = np.linalg.eigh(symmetric)
        return energies, orthogonalizer @ coefficients_prime, coefficients_prime

    energies, coefficients, coefficients_prime = diagonalize(
        orthogonalizer.T @ core @ orthogonalizer
    )
    density = density_from_coefficients(coefficients, n_occupied)

    fock_history: list[np.ndarray] = []
    error_history: list[np.ndarray] = []
    history: list[ScfHistory] = []
    previous_energy = 0.0
    converged = False
    iterations = 0
    level_shift_active = False

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration
        fock = build_fock(core, density, eri)
        energy = electronic_energy(density, core, fock) + v_nuc
        energy_change = energy - previous_energy

        fock_prime = orthogonalizer.T @ fock @ orthogonalizer
        # Плотность в ортогональном базисе — это D' = 2 C'_occ C'_occ^T, а не
        # X^T D X: при C = X C' верно D = X D' X^T, поэтому обратное
        # преобразование требует X^{-1}, а не X. Ошибка здесь не ломает энергию,
        # но навсегда останавливает DIIS: его невязка перестаёт стремиться к нулю.
        occupied_prime = coefficients_prime[:, :n_occupied]
        density_prime = 2.0 * occupied_prime @ occupied_prime.T
        commutator = density_prime @ fock_prime - fock_prime @ density_prime
        diis_error = float(np.max(np.abs(commutator)))

        strategy = "plain"
        effective_fock_prime = fock_prime

        # --- DIIS: экстраполяция фокиана, а не плотности ------------------
        if iteration >= config.diis_start:
            fock_history.append(fock_prime)
            error_history.append(commutator)
            if len(fock_history) > config.diis_space:
                fock_history.pop(0)
                error_history.pop(0)
            extrapolated = _diis_extrapolate(fock_history, error_history)
            if extrapolated is not None:
                effective_fock_prime = extrapolated
                strategy = "diis"
                if "diis" not in strategies:
                    strategies.append("diis")

        # --- сдвиг уровней: только если DIIS застопорился ------------------
        stalled = (
            len(history) >= config.diis_space + config.diis_start
            and diis_error > config.stall_ratio * history[-config.diis_space].diis_error
            and diis_error > config.density_tolerance
        )
        if stalled and not level_shift_active:
            level_shift_active = True
            strategies.append("level-shift")
        if level_shift_active:
            virtual_projector = np.eye(fock_prime.shape[0]) - occupied_prime @ occupied_prime.T
            effective_fock_prime = fock_prime + config.level_shift * virtual_projector
            strategy = "level-shift"

        energies, coefficients, coefficients_prime = diagonalize(effective_fock_prime)
        new_density = density_from_coefficients(coefficients, n_occupied)

        # --- гашение: реактивное, а не «на всякий случай» -----------------
        # Шаги 1..damping_rounds гасятся безусловно (по умолчанию их нет).
        # Кроме того, гашение включается, когда предыдущий шаг повысил энергию.
        # В обоих случаях буферы DIIS сбрасываются: векторы из другой
        # траектории экстраполировать нельзя — именно это и останавливало
        # сходимость, пока гашение стояло на первых итерациях всегда.
        regressed = iteration > 1 and energy_change > 0.0
        if iteration <= config.damping_rounds or (regressed and not level_shift_active):
            new_density = (
                config.damping_factor * density + (1.0 - config.damping_factor) * new_density
            )
            fock_history.clear()
            error_history.clear()
            strategy = "damping"
            if "damping" not in strategies:
                strategies.append("damping")

        density_change = float(np.max(np.abs(new_density - density)))
        history.append(
            ScfHistory(
                iteration=iteration,
                energy=energy,
                energy_change=energy_change,
                density_change=density_change,
                diis_error=diis_error,
                strategy=strategy,
            )
        )
        previous_energy = energy
        density = new_density

        if (
            iteration > 1
            and abs(energy_change) < config.energy_tolerance
            and diis_error < config.density_tolerance
        ):
            converged = True
            break

    fock = build_fock(core, density, eri)
    total = electronic_energy(density, core, fock) + v_nuc
    return ScfResult(
        total_energy=total,
        electronic_energy=total - v_nuc,
        nuclear_repulsion=v_nuc,
        orbital_energies=tuple(float(value) for value in energies),
        coefficients=coefficients,
        density=density,
        converged=converged,
        iterations=iterations,
        history=history,
        strategies_used=tuple(strategies),
        elapsed_seconds=time.perf_counter() - started,
    )
