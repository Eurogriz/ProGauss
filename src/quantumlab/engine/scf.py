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

Что сознательно не реализовано в этом срезе: EDIIS, SOSCF, проверка
устойчивости волновой функции, дробные занятия. Они заявлены в реестре как
``not_implemented`` **и отклоняются при валидации спецификации**: запрос
«посчитай с EDIIS» получает штатную ошибку, а не расчёт, молча выполненный
без него.
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
    #: Порог невязки, при котором сдвиг уровней выключается.
    #:
    #: Сдвиг — временный стабилизатор, а не постоянное состояние: сдвинутый
    #: фокиан ``F + λP_вирт`` имеет другую стационарную точку, поэтому при
    #: постоянно включённом сдвиге условие стационарности исходной задачи не
    #: выполняется никогда. До этого исправления SCF на радикале CH/STO-3G
    #: сходился по энергии до 1e-14 и при этом не проходил по невязке вообще.
    #:
    #: Значение выбрано измерением на CH/STO-3G (жёсткий допуск 1e-9):
    #: 1e-2 → 44 итерации, 1e-3 → 77, 1e-4 → 133, 1e-5 → не сходится за 300.
    #: Замкнутой оболочки порог не касается: там сдвиг не включается.
    level_shift_release: float = 1e-2


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


def coulomb_matrix(density: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Кулоновская матрица ``J_μν = Σ_λσ D_λσ (μν|λσ)``.

    В UHF кулоновский член строится по **полной** плотности (α + β): электрон
    любого спина отталкивается от полного заряда.
    """
    return np.asarray(np.einsum("ls,uvls->uv", density, eri, optimize=True))


def exchange_matrix(density: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Обменная матрица ``K_μν = Σ_λσ D_λσ (μλ|νσ)``.

    В UHF обмен действует только между электронами **одного** спина, поэтому
    K строится по плотности соответствующего спинового канала.
    """
    return np.asarray(np.einsum("ls,ulvs->uv", density, eri, optimize=True))


def build_fock(core: np.ndarray, density: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Фок-матрица ``F = H + J − ½K``.

    Коэффициент ½ у обменного члена — следствие RHF (двойное занятие).
    """
    return np.asarray(core + coulomb_matrix(density, eri) - 0.5 * exchange_matrix(density, eri))


def solve_roothaan(fock: np.ndarray, orthogonalizer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Решает уравнение Рутана в ортогонализованном базисе.

    Возвращает собственные значения и коэффициенты в исходном (неортогональном)
    базисе: ``C = X C'``.
    """
    transformed = orthogonalizer.T @ fock @ orthogonalizer
    transformed = 0.5 * (transformed + transformed.T)
    energies, coefficients_prime = np.linalg.eigh(transformed)
    return energies, orthogonalizer @ coefficients_prime


def density_from_coefficients(
    coefficients: np.ndarray, n_occupied: int, occupation: float = 2.0
) -> np.ndarray:
    """Плотность ``D = occupation · C_occ C_occ^T``.

    Для RHF занятие равно 2 (каждая МО занята парой), для спинового канала UHF
    — 1: там орбиталь занята одним электроном.
    """
    occupied = coefficients[:, :n_occupied]
    return np.asarray(occupation * occupied @ occupied.T)


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


@dataclass(frozen=True, slots=True)
class PrecomputedIntegrals:
    """Одноэлектронные и двухэлектронные интегралы, собранные заранее.

    Нужны, чтобы вызывающая сторона могла отнести стоимость интегралов к своему
    этапу. Без этого сборка тензора ERI — самая дорогая часть расчёта —
    попадала в этап ``scf`` и отчёт о временах этапов вводил в заблуждение:
    на воде/6-31G этап ``integrals`` показывал 0.012 с, тогда как двухэлектронные
    интегралы стоили около секунды.
    """

    overlap: np.ndarray
    core: np.ndarray
    eri: np.ndarray


def build_integrals(basis: BasisSet, molecule: Molecule) -> PrecomputedIntegrals:
    """Собирает все интегралы, нужные RHF."""
    return PrecomputedIntegrals(
        overlap=build_overlap(basis, molecule),
        core=build_core_hamiltonian(basis, molecule),
        eri=build_electron_repulsion(basis, molecule),
    )


def run_rhf(
    basis: BasisSet,
    molecule: Molecule,
    settings: ScfSettings | None = None,
    *,
    integrals: PrecomputedIntegrals | None = None,
) -> ScfResult:
    """Выполняет RHF-расчёт.

    Число занятых орбиталей берётся из числа электронов молекулы; расчёт
    предполагает замкнутую оболочку. Нечётное число электронов — ошибка
    вызывающей стороны (для неё нужны ROHF/UHF, которых в этом срезе нет).

    На каждой итерации выполняется ровно одна диагонализация: фокиан строится
    по текущей плотности, при необходимости экстраполируется (DIIS) и/или
    сдвигается по уровням, затем диагонализуется в ортогональном базисе.

    Интегралы можно передать готовыми через ``integrals`` — тогда их стоимость
    относится к этапу вызывающей стороны, а не к ``scf``.
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

    prepared = integrals if integrals is not None else build_integrals(basis, molecule)
    overlap = prepared.overlap
    core = prepared.core
    eri = prepared.eri
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
        # Сдвиг уровней выключается, как только невязка достаточно мала: дальше
        # DIIS доводит решение сам, уже по невозмущённому фокиану. Буферы DIIS
        # сбрасываются — векторы из сдвинутой траектории экстраполировать нельзя.
        if level_shift_active and diis_error < config.level_shift_release:
            level_shift_active = False
            fock_history.clear()
            error_history.clear()
            strategies.append("level-shift-released")

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


# --------------------------------------------------------------------------- #
# UHF: неограниченный по спину метод Хартри–Фока
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class UhfResult:
    """Результат UHF — два независимых спиновых канала.

    В отличие от RHF орбитали α и β разные, поэтому у результата две плотности
    и два набора энергий орбиталей. ``s_squared`` — ожидание ⟨Ŝ²⟩: по нему
    видно спиновое загрязнение, которое в UHF возможно и о котором нельзя
    молчать (§54 ТЗ).
    """

    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    alpha_energies: tuple[float, ...]
    beta_energies: tuple[float, ...]
    alpha_coefficients: np.ndarray
    beta_coefficients: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    s_squared: float
    converged: bool
    iterations: int
    history: list[ScfHistory] = field(default_factory=list)
    strategies_used: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0


def spin_contamination(
    alpha_coefficients: np.ndarray,
    beta_coefficients: np.ndarray,
    n_alpha: int,
    n_beta: int,
    overlap: np.ndarray,
) -> float:
    """Ожидание ⟨Ŝ²⟩ для однодетерминантной UHF-волновой функции.

    ``⟨Ŝ²⟩ = S_z(S_z+1) + n_β − Σ_ij |⟨φ_i^α|φ_j^β⟩|²``.

    Коэффициенты заданы в неортогональном базисе (``C^T S C = I``), поэтому
    скалярное произведение орбиталей — это ``C_α^T S C_β``, а не ``C_α^T C_β``.
    Без матрицы перекрытий даже совпадающие каналы дали бы ненулевое ⟨Ŝ²⟩.

    Для чистого спинового состояния значение равно ``S(S+1)``; превышение —
    спиновое загрязнение. У RHF такой величины нет по построению, поэтому она
    сообщается только для UHF.
    """
    s_z = 0.5 * (n_alpha - n_beta)
    occupied_alpha = alpha_coefficients[:, :n_alpha]
    occupied_beta = beta_coefficients[:, :n_beta]
    orbital_overlap = occupied_alpha.T @ overlap @ occupied_beta
    return float(s_z * (s_z + 1.0) + n_beta - np.sum(orbital_overlap**2))


def spin_population(electrons: int, multiplicity: int) -> tuple[int, int]:
    """Числа электронов в каналах α и β по заряду и мультиплетности.

    Невозможное сочетание (нечётное число неспаренных при чётном числе
    электронов, либо больше неспаренных, чем электронов) — ошибка вызывающей
    стороны: домен её уже отловил, но движок не имеет права считать дальше.
    """
    unpaired = multiplicity - 1
    if unpaired > electrons or (electrons - unpaired) % 2 != 0:
        msg = f"Мультиплетность {multiplicity} несовместима с числом электронов {electrons}"
        raise ValueError(msg)
    return (electrons + unpaired) // 2, (electrons - unpaired) // 2


def run_uhf(
    basis: BasisSet,
    molecule: Molecule,
    settings: ScfSettings | None = None,
    *,
    integrals: PrecomputedIntegrals | None = None,
) -> UhfResult:
    """Выполняет UHF-расчёт.

    Отличия от RHF: плотности и фокианы ведутся отдельно для α и β, кулоновский
    член строится по полной плотности, а обменный — только по плотности своего
    канала. Интегралы при этом ровно те же, что и в RHF: новых интегральных
    выражений UHF не требует.

    Стартовая догадка — одинаковые α и β из ядра Гамильтона, поэтому для
    замкнутой оболочки UHF сходится к RHF-решению (и обязан дать ту же энергию).
    """
    config = settings or ScfSettings()
    started = time.perf_counter()
    strategies: list[str] = ["core-hamiltonian-guess"]

    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)

    prepared = integrals if integrals is not None else build_integrals(basis, molecule)
    overlap, core, eri = prepared.overlap, prepared.core, prepared.eri
    v_nuc = nuclear_repulsion(molecule)
    orthogonalizer = canonical_orthogonalizer(overlap)

    def diagonalize(fock_prime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symmetric = 0.5 * (fock_prime + fock_prime.T)
        energies, coefficients_prime = np.linalg.eigh(symmetric)
        return energies, orthogonalizer @ coefficients_prime, coefficients_prime

    alpha_energies, alpha_coefficients, alpha_prime = diagonalize(
        orthogonalizer.T @ core @ orthogonalizer
    )
    beta_energies, beta_coefficients, beta_prime = (
        alpha_energies,
        alpha_coefficients,
        alpha_prime,
    )
    density_alpha = density_from_coefficients(alpha_coefficients, n_alpha, occupation=1.0)
    density_beta = density_from_coefficients(beta_coefficients, n_beta, occupation=1.0)

    diis_alpha: list[np.ndarray] = []
    diis_beta: list[np.ndarray] = []
    error_alpha: list[np.ndarray] = []
    error_beta: list[np.ndarray] = []
    history: list[ScfHistory] = []
    previous_energy = 0.0
    converged = False
    iterations = 0
    level_shift_active = False

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration
        density_total = density_alpha + density_beta
        coulomb = coulomb_matrix(density_total, eri)
        fock_alpha = core + coulomb - exchange_matrix(density_alpha, eri)
        fock_beta = core + coulomb - exchange_matrix(density_beta, eri)
        energy = (
            float(
                0.5
                * (
                    np.sum(density_total * core)
                    + np.sum(density_alpha * fock_alpha)
                    + np.sum(density_beta * fock_beta)
                )
            )
            + v_nuc
        )
        energy_change = energy - previous_energy

        fock_alpha_prime = orthogonalizer.T @ fock_alpha @ orthogonalizer
        fock_beta_prime = orthogonalizer.T @ fock_beta @ orthogonalizer
        # Невязка считается по каждому каналу в ортогональном базисе; критерий
        # сходимости — худший из двух: сходимость только α не означает
        # сходимость расчёта.
        occupied_alpha = alpha_prime[:, :n_alpha]
        occupied_beta = beta_prime[:, :n_beta]
        # Коммутатор считается с плотностью в ортогональном базисе, а не с
        # матрицей занятых орбиталей: [D', F'], где D' = C'_occ C'_occ^T
        # (занятие 1 — каналы разделены). Тот же переход, что и в RHF.
        density_alpha_prime = occupied_alpha @ occupied_alpha.T
        density_beta_prime = occupied_beta @ occupied_beta.T
        residual_alpha = density_alpha_prime @ fock_alpha_prime - fock_alpha_prime @ (
            density_alpha_prime
        )
        residual_beta = density_beta_prime @ fock_beta_prime - fock_beta_prime @ (
            density_beta_prime
        )
        diis_error = float(max(np.max(np.abs(residual_alpha)), np.max(np.abs(residual_beta))))

        strategy = "plain"
        effective_alpha = fock_alpha_prime
        effective_beta = fock_beta_prime

        if iteration >= config.diis_start:
            diis_alpha.append(fock_alpha_prime)
            diis_beta.append(fock_beta_prime)
            error_alpha.append(residual_alpha)
            error_beta.append(residual_beta)
            if len(diis_alpha) > config.diis_space:
                diis_alpha.pop(0)
                diis_beta.pop(0)
                error_alpha.pop(0)
                error_beta.pop(0)
            extrapolated_alpha = _diis_extrapolate(diis_alpha, error_alpha)
            extrapolated_beta = _diis_extrapolate(diis_beta, error_beta)
            if extrapolated_alpha is not None and extrapolated_beta is not None:
                effective_alpha = extrapolated_alpha
                effective_beta = extrapolated_beta
                strategy = "diis"
                if "diis" not in strategies:
                    strategies.append("diis")

        # Сдвиг уровней — временный стабилизатор: сдвинутый фокиан имеет другую
        # стационарную точку, поэтому держать его до конца нельзя (§10 ТЗ).
        if level_shift_active and diis_error < config.level_shift_release:
            level_shift_active = False
            diis_alpha.clear()
            diis_beta.clear()
            error_alpha.clear()
            error_beta.clear()
            strategies.append("level-shift-released")

        stalled = (
            len(history) >= config.diis_space + config.diis_start
            and diis_error > config.stall_ratio * history[-config.diis_space].diis_error
            and diis_error > config.density_tolerance
        )
        if stalled and not level_shift_active:
            level_shift_active = True
            strategies.append("level-shift")
        if level_shift_active:
            identity = np.eye(fock_alpha_prime.shape[0])
            effective_alpha = effective_alpha + config.level_shift * (
                identity - occupied_alpha @ occupied_alpha.T
            )
            effective_beta = effective_beta + config.level_shift * (
                identity - occupied_beta @ occupied_beta.T
            )
            strategy = "level-shift"

        alpha_energies, alpha_coefficients, alpha_prime = diagonalize(effective_alpha)
        beta_energies, beta_coefficients, beta_prime = diagonalize(effective_beta)
        new_alpha = density_from_coefficients(alpha_coefficients, n_alpha, occupation=1.0)
        new_beta = density_from_coefficients(beta_coefficients, n_beta, occupation=1.0)

        regressed = iteration > 1 and energy_change > 0.0
        if iteration <= config.damping_rounds or (regressed and not level_shift_active):
            new_alpha = (
                config.damping_factor * density_alpha + (1.0 - config.damping_factor) * new_alpha
            )
            new_beta = config.damping_factor * density_beta + (1.0 - config.damping_factor) * (
                new_beta
            )
            diis_alpha.clear()
            diis_beta.clear()
            error_alpha.clear()
            error_beta.clear()
            strategy = "damping"
            if "damping" not in strategies:
                strategies.append("damping")

        density_change = float(
            max(np.max(np.abs(new_alpha - density_alpha)), np.max(np.abs(new_beta - density_beta)))
        )
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
        density_alpha = new_alpha
        density_beta = new_beta

        if (
            iteration > 1
            and abs(energy_change) < config.energy_tolerance
            and diis_error < config.density_tolerance
        ):
            converged = True
            break

    density_total = density_alpha + density_beta
    coulomb = coulomb_matrix(density_total, eri)
    fock_alpha = core + coulomb - exchange_matrix(density_alpha, eri)
    fock_beta = core + coulomb - exchange_matrix(density_beta, eri)
    total = (
        float(
            0.5
            * (
                np.sum(density_total * core)
                + np.sum(density_alpha * fock_alpha)
                + np.sum(density_beta * fock_beta)
            )
        )
        + v_nuc
    )
    return UhfResult(
        total_energy=total,
        electronic_energy=total - v_nuc,
        nuclear_repulsion=v_nuc,
        alpha_energies=tuple(float(value) for value in alpha_energies),
        beta_energies=tuple(float(value) for value in beta_energies),
        alpha_coefficients=alpha_coefficients,
        beta_coefficients=beta_coefficients,
        density_alpha=density_alpha,
        density_beta=density_beta,
        s_squared=spin_contamination(
            alpha_coefficients, beta_coefficients, n_alpha, n_beta, overlap
        ),
        converged=converged,
        iterations=iterations,
        history=history,
        strategies_used=tuple(strategies),
        elapsed_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class RohfResult:
    """Результат ROHF — ограниченный открытый оболочечный метод.

    В отличие от UHF орбитали α и β общие: различаются только занятия
    (``n_alpha`` и ``n_beta`` первых столбцов одного и того же ``C``). Поэтому
    волновая функция остаётся собственной функцией Ŝ², и ``s_squared`` обязан
    равняться ``S(S+1)`` точно — без спинового загрязнения, возможного в UHF.
    """

    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    orbital_energies: tuple[float, ...]
    coefficients: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    s_squared: float
    converged: bool
    iterations: int
    history: list[ScfHistory] = field(default_factory=list)
    strategies_used: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    @property
    def alpha_energies(self) -> tuple[float, ...]:
        """Собственные значения эффективного фокиана Рутаана.

        У ROHF орбитали общие, поэтому набор один, и каналы α/β дают одинаковые
        числа. Отдельных «орбитальных энергий α и β» здесь нет: эффективный
        оператор не воспроизводит их по отдельности, и выдавать один набор под
        двумя именами было бы неправдой.
        """
        return self.orbital_energies

    @property
    def beta_energies(self) -> tuple[float, ...]:
        """Тот же набор, что и :attr:`alpha_energies` — орбитали общие."""
        return self.orbital_energies


def roothaan_effective_fock(
    fock_alpha: np.ndarray,
    fock_beta: np.ndarray,
    density_alpha: np.ndarray,
    density_beta: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    """Эффективный фокиан Рутаана (Roothaan, 1960).

    Обычные спиновые фокианы ``Fα`` и ``Fβ`` нельзя диагонализировать одним
    преобразованием: у открытой оболочки разные блоки требуют разного оператора.
    Рутаан строит один эрмитов оператор, блочная структура которого такова:

    ============  ========  ======  =========
    пространство  closed    open    virtual
    ============  ========  ======  =========
    closed        Fc        Fb      Fc
    open          Fb        Fc      Fa
    virtual       Fc        Fa      Fc
    ============  ========  ======  =========

    где ``Fc = (Fα + Fβ)/2``. Проекторы строятся по плотностям: ``Dβ`` — это
    плотность замкнутой оболочки, ``Dα − Dβ`` — открытой, ``I − Dα`` —
    виртуального пространства.

    Проверка предельного перехода: при ``n_α = n_β`` открытый проектор зануляется,
    ``Fc = Fα = Fβ``, и поскольку ``Pc + Pv = I``, выражение сворачивается ровно
    в ``Fc`` — то есть ROHF замкнутой оболочки совпадает с RHF.
    """
    fock_closed = 0.5 * (fock_alpha + fock_beta)
    projector_closed = density_beta @ overlap
    projector_open = (density_alpha - density_beta) @ overlap
    projector_virtual = np.eye(overlap.shape[0]) - density_alpha @ overlap

    fock: np.ndarray = 0.5 * (projector_closed.T @ fock_closed @ projector_closed)
    fock += 0.5 * (projector_open.T @ fock_closed @ projector_open)
    fock += 0.5 * (projector_virtual.T @ fock_closed @ projector_virtual)
    fock += projector_open.T @ fock_beta @ projector_closed
    fock += projector_open.T @ fock_alpha @ projector_virtual
    fock += projector_virtual.T @ fock_closed @ projector_closed
    # Складываем с транспонированной: перечисленные слагаемые заполняют только
    # половину блоков, а оператор обязан быть эрмитовым.
    return np.asarray(fock + fock.T)


def run_rohf(
    basis: BasisSet,
    molecule: Molecule,
    settings: ScfSettings | None = None,
    *,
    integrals: PrecomputedIntegrals | None = None,
) -> RohfResult:
    """Выполняет ROHF-расчёт.

    Энергия выражается той же формулой, что и в UHF, — различие не в функционале
    энергии, а в допустимых плотностях: ROHF требует общих орбиталей для обоих
    каналов. Поэтому α- и β-плотности строятся из одного ``C`` с разным числом
    занятых столбцов.

    Диагонализуется эффективный фокиан Рутаана, а не ``Fα``: собственные значения
    ``Fα`` не соответствуют орбитальным энергиям открытой оболочки.
    """
    config = settings or ScfSettings()
    started = time.perf_counter()
    strategies: list[str] = ["core-hamiltonian-guess"]

    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)

    prepared = integrals if integrals is not None else build_integrals(basis, molecule)
    overlap, core, eri = prepared.overlap, prepared.core, prepared.eri
    v_nuc = nuclear_repulsion(molecule)
    orthogonalizer = canonical_orthogonalizer(overlap)

    def diagonalize(fock_prime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symmetric = 0.5 * (fock_prime + fock_prime.T)
        energies, coefficients_prime = np.linalg.eigh(symmetric)
        return energies, orthogonalizer @ coefficients_prime, coefficients_prime

    energies, coefficients, coefficients_prime = diagonalize(
        orthogonalizer.T @ core @ orthogonalizer
    )
    density_alpha = density_from_coefficients(coefficients, n_alpha, occupation=1.0)
    density_beta = density_from_coefficients(coefficients, n_beta, occupation=1.0)

    diis_focks: list[np.ndarray] = []
    diis_errors: list[np.ndarray] = []
    history: list[ScfHistory] = []
    previous_energy = 0.0
    converged = False
    iterations = 0
    level_shift_active = False

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration
        density_total = density_alpha + density_beta
        coulomb = coulomb_matrix(density_total, eri)
        fock_alpha = core + coulomb - exchange_matrix(density_alpha, eri)
        fock_beta = core + coulomb - exchange_matrix(density_beta, eri)
        energy = (
            float(
                0.5
                * (
                    np.sum(density_total * core)
                    + np.sum(density_alpha * fock_alpha)
                    + np.sum(density_beta * fock_beta)
                )
            )
            + v_nuc
        )
        energy_change = energy - previous_energy

        effective = roothaan_effective_fock(
            fock_alpha, fock_beta, density_alpha, density_beta, overlap
        )
        effective_prime = orthogonalizer.T @ effective @ orthogonalizer

        # Занятое пространство в ROHF одно на оба канала: это первые n_alpha
        # столбцов общего C, то есть проектор Dα в ортогональном базисе.
        occupied = coefficients_prime[:, :n_alpha]
        occupied_projector = occupied @ occupied.T
        residual = occupied_projector @ effective_prime - effective_prime @ occupied_projector
        diis_error = float(np.max(np.abs(residual)))

        strategy = "plain"
        effective_used = effective_prime
        if iteration >= config.diis_start:
            diis_focks.append(effective_prime)
            diis_errors.append(residual)
            if len(diis_focks) > config.diis_space:
                diis_focks.pop(0)
                diis_errors.pop(0)
            extrapolated = _diis_extrapolate(diis_focks, diis_errors)
            if extrapolated is not None:
                effective_used = extrapolated
                strategy = "diis"
                if "diis" not in strategies:
                    strategies.append("diis")

        if level_shift_active and diis_error < config.level_shift_release:
            level_shift_active = False
            diis_focks.clear()
            diis_errors.clear()
            strategies.append("level-shift-released")

        stalled = (
            len(history) >= config.diis_space + config.diis_start
            and diis_error > config.stall_ratio * history[-config.diis_space].diis_error
            and diis_error > config.density_tolerance
        )
        if stalled and not level_shift_active:
            level_shift_active = True
            strategies.append("level-shift")
        if level_shift_active:
            identity = np.eye(effective_prime.shape[0])
            effective_used = effective_used + config.level_shift * (identity - occupied_projector)

        energies, coefficients, coefficients_prime = diagonalize(effective_used)
        density_alpha = density_from_coefficients(coefficients, n_alpha, occupation=1.0)
        density_beta = density_from_coefficients(coefficients, n_beta, occupation=1.0)

        history.append(
            ScfHistory(
                iteration=iteration,
                energy=energy,
                energy_change=energy_change,
                density_change=0.0,
                diis_error=diis_error,
                strategy=strategy,
            )
        )
        previous_energy = energy
        if abs(energy_change) < config.energy_tolerance and diis_error < config.density_tolerance:
            converged = True
            break

    density_total = density_alpha + density_beta
    coulomb = coulomb_matrix(density_total, eri)
    fock_alpha = core + coulomb - exchange_matrix(density_alpha, eri)
    fock_beta = core + coulomb - exchange_matrix(density_beta, eri)
    electronic = float(
        0.5
        * (
            np.sum(density_total * core)
            + np.sum(density_alpha * fock_alpha)
            + np.sum(density_beta * fock_beta)
        )
    )
    return RohfResult(
        total_energy=electronic + v_nuc,
        electronic_energy=electronic,
        nuclear_repulsion=v_nuc,
        orbital_energies=tuple(float(value) for value in energies),
        coefficients=coefficients,
        density_alpha=density_alpha,
        density_beta=density_beta,
        # Тот же общий определитель, что и в UHF. Для ROHF он обязан дать ровно
        # S(S+1): замкнутые орбитали совпадают в обоих каналах, и сумма квадратов
        # перекрытий в точности сокращает n_beta.
        s_squared=spin_contamination(coefficients, coefficients, n_alpha, n_beta, overlap),
        converged=converged,
        iterations=iterations,
        history=history,
        strategies_used=tuple(strategies),
        elapsed_seconds=time.perf_counter() - started,
    )
