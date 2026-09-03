"""Решатель Кона–Шэма в приближении LDA.

Отличия от RHF сводятся к одному слагаемому в фокиане: вместо обменного члена
``−½K`` (или вместе с его долей у гибридов) добавляется обменно-корреляционный
потенциал, собранный численно на квадратурной сетке:

``V_xc[μν] = Σ_g w_g v_xc(ρ_g) φ_μ(r_g) φ_ν(r_g)``

Остальной итерационный процесс — DIIS, гашение, сдвиг уровней — тот же, что в
RHF, и намеренно не переписан: две независимые реализации сходимости означали бы
два разных поведения для одного и того же расчёта.

Число электронов делится поровну (замкнутая оболочка), поэтому решатель
называется RKS. UKS требует отдельных плотностей для каналов, как и UHF.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import GridPreset
from quantumlab.engine.basis import BasisSet, nuclear_repulsion
from quantumlab.engine.contracts import ExchangeCorrelationFunctional
from quantumlab.engine.functional import (
    density_at_points,
    density_gradient_at_points,
    evaluate_basis_with_gradients,
)
from quantumlab.engine.quadrature import QuadratureGrid, build_grid
from quantumlab.engine.scf import (
    PrecomputedIntegrals,
    ScfHistory,
    ScfResult,
    ScfSettings,
    UhfResult,
    _diis_extrapolate,
    build_integrals,
    canonical_orthogonalizer,
    coulomb_matrix,
    density_from_coefficients,
    exchange_matrix,
    spin_contamination,
    spin_population,
)


@dataclass(slots=True)
class RksResult:
    """Результат RKS — тот же набор полей, что у RHF, плюс энергия XC."""

    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    xc_energy: float
    orbital_energies: tuple[float, ...]
    coefficients: np.ndarray
    density: np.ndarray
    converged: bool
    iterations: int
    grid_points: int
    #: Доля точного обмена α. Хранится в результате, потому что энергия
    #: гибрида без неё неоднозначна: одно и то же число можно прочесть и как
    #: результат с α = 0, и как с α = 0.25. Проверки качества восстанавливают по
    #: ней обменный член −¼α·D:K.
    exact_exchange_fraction: float = 0.0
    #: Обменно-корреляционный потенциал на сошедшейся плотности. Хранится в
    #: результате, чтобы проверки качества могли восстановить настоящий фокиан
    #: RKS, не пересобирая сетку и не считая XC второй раз.
    v_xc: np.ndarray | None = None
    history: list[ScfHistory] = field(default_factory=list)
    strategies_used: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    def as_scf_result(self) -> ScfResult:
        """Приводит к общему виду SCF.

        Нужно вызывающей стороне, которая работает с результатом, не зная,
        каким методом он получен; энергия XC при этом не теряется — она лежит
        в ``RksResult`` и попадает в отчёт отдельно.
        """
        return ScfResult(
            total_energy=self.total_energy,
            electronic_energy=self.electronic_energy,
            nuclear_repulsion=self.nuclear_repulsion,
            orbital_energies=self.orbital_energies,
            coefficients=self.coefficients,
            density=self.density,
            converged=self.converged,
            iterations=self.iterations,
            history=self.history,
            strategies_used=self.strategies_used,
            elapsed_seconds=self.elapsed_seconds,
        )


@dataclass(slots=True)
class UksResult:
    """Результат UKS — спиново-разделённая версия RKS.

    Плотности, энергии и коэффициенты орбиталей ведутся по каналам α и β
    (как в UHF); обменно-корреляционный член — ``xc_energy`` и доля точного
    обмена. ``v_xc_alpha``/``v_xc_beta`` — потенциалы каналов на сошедшейся
    плотности: проверки качества восстанавливают по ним фокианы, не пересчитывая
    сетку.
    """

    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    xc_energy: float
    alpha_energies: tuple[float, ...]
    beta_energies: tuple[float, ...]
    alpha_coefficients: np.ndarray
    beta_coefficients: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    s_squared: float
    converged: bool
    iterations: int
    grid_points: int
    #: Доля точного обмена α (0.25 для PBE0, 0.20 для B3LYP, 0 для чистых).
    exact_exchange_fraction: float = 0.0
    #: V_xc^α и V_xc^β на сошедшейся плотности (см. docstring класса).
    v_xc_alpha: np.ndarray | None = None
    v_xc_beta: np.ndarray | None = None
    history: list[ScfHistory] = field(default_factory=list)
    strategies_used: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    def as_uhf_result(self) -> UhfResult:
        """Приводит к виду UHF для переиспользования свойств открытой оболочки.

        Спин-орбитальные энергии, плотности и ⟨Ŝ²⟩ — те же величины, что и в
        UHF; энергия XC не теряется, она лежит в ``UksResult`` и попадает в
        отчёт отдельно.
        """
        return UhfResult(
            total_energy=self.total_energy,
            electronic_energy=self.electronic_energy,
            nuclear_repulsion=self.nuclear_repulsion,
            alpha_energies=self.alpha_energies,
            beta_energies=self.beta_energies,
            alpha_coefficients=self.alpha_coefficients,
            beta_coefficients=self.beta_coefficients,
            density_alpha=self.density_alpha,
            density_beta=self.density_beta,
            s_squared=self.s_squared,
            converged=self.converged,
            iterations=self.iterations,
            history=list(self.history),
            strategies_used=self.strategies_used,
            elapsed_seconds=self.elapsed_seconds,
        )


@dataclass(slots=True)
class UksResult:
    """Результат UKS — спиново-разделённая версия RKS.

    Плотности, энергии и коэффициенты орбиталей ведутся по каналам α и β
    (как в UHF); обменно-корреляционный член — ``xc_energy`` и доля точного
    обмена. ``v_xc_alpha``/``v_xc_beta`` — потенциалы каналов на сошедшейся
    плотности: проверки качества восстанавливают по ним фокианы, не пересчитывая
    сетку.
    """

    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    xc_energy: float
    alpha_energies: tuple[float, ...]
    beta_energies: tuple[float, ...]
    alpha_coefficients: np.ndarray
    beta_coefficients: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    s_squared: float
    converged: bool
    iterations: int
    grid_points: int
    #: Доля точного обмена α (0.25 для PBE0, 0.20 для B3LYP, 0 для чистых).
    exact_exchange_fraction: float = 0.0
    #: V_xc^α и V_xc^β на сошедшейся плотности (см. docstring класса).
    v_xc_alpha: np.ndarray | None = None
    v_xc_beta: np.ndarray | None = None
    history: list[ScfHistory] = field(default_factory=list)
    strategies_used: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    def as_uhf_result(self) -> UhfResult:
        """Приводит к виду UHF для переиспользования свойств открытой оболочки.

        Спин-орбитальные энергии, плотности и ⟨Ŝ²⟩ — те же величины, что и в
        UHF; энергия XC не теряется, она лежит в ``UksResult`` и попадает в
        отчёт отдельно.
        """
        return UhfResult(
            total_energy=self.total_energy,
            electronic_energy=self.electronic_energy,
            nuclear_repulsion=self.nuclear_repulsion,
            alpha_energies=self.alpha_energies,
            beta_energies=self.beta_energies,
            alpha_coefficients=self.alpha_coefficients,
            beta_coefficients=self.beta_coefficients,
            density_alpha=self.density_alpha,
            density_beta=self.density_beta,
            s_squared=self.s_squared,
            converged=self.converged,
            iterations=self.iterations,
            history=list(self.history),
            strategies_used=self.strategies_used,
            elapsed_seconds=self.elapsed_seconds,
        )


def xc_matrix_and_energy(
    grid: QuadratureGrid,
    values: np.ndarray,
    gradients: np.ndarray,
    rho: np.ndarray,
    density_gradient: np.ndarray,
    functional: ExchangeCorrelationFunctional,
) -> tuple[np.ndarray, float]:
    """Обменно-корреляционная матрица и энергия на текущей плотности.

    Возвращает ``(V_xc, E_xc)``. Энергия считается как ``Σ_g w_g ρ_g ε_xc``,
    а не через след ``D·V_xc``: для нелинейного функционала это разные величины,
    и верна именно первая.

    Для LDA достаточно одного слагаемого ``v_ρ φ_μ φ_ν``. GGA зависит ещё и от
    ``σ = |∇ρ|²``, и вариация по матрице плотности даёт второй член:

    ``V_xc[μν] = Σ_g w_g [ v_ρ φ_μ φ_ν + 2 v_σ ∇ρ·(∇φ_μ φ_ν + φ_μ ∇φ_ν) ]``

    Множитель 2 здесь не «коэффициент двойного счёта», а ``∂σ/∂∇ρ = 2∇ρ``;
    перепутать их легко, а ошибка не видна ни по сходимости SCF, ни по
    коммутатору — только по энергии.
    """
    xc = functional.evaluate(grid.points, rho, density_gradient)
    energy = float(np.sum(grid.weights * rho * xc.energy_density))

    scaled_rho = grid.weights * xc.vrho
    matrix = np.einsum("p,pg,ph->gh", scaled_rho, values, values, optimize=True)

    if xc.vsigma is not None:
        # ∇φ_μ · ∇ρ в каждой точке: свёртка по декартовой оси.
        gradient_projection = np.einsum("pgd,pd->pg", gradients, density_gradient, optimize=True)
        scaled_sigma = 2.0 * grid.weights * xc.vsigma
        matrix += np.einsum("p,pg,ph->gh", scaled_sigma, gradient_projection, values, optimize=True)
        matrix += np.einsum("p,pg,ph->gh", scaled_sigma, values, gradient_projection, optimize=True)
    return np.asarray(matrix), energy


def xc_matrix_and_energy_spin(
    grid: QuadratureGrid,
    values: np.ndarray,
    gradients: np.ndarray,
    rho_alpha: np.ndarray,
    rho_beta: np.ndarray,
    grad_alpha: np.ndarray,
    grad_beta: np.ndarray,
    functional: ExchangeCorrelationFunctional,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Спиновые обменно-корреляционные матрицы и энергия.

    Возвращает ``(V_xc^α, V_xc^β, E_xc)``. Потенциал канала α — производная
    энергии по плотности канала α при фиксированной β:

    ``V_xc^α[μν] = Σ_g w_g [ v_ρ^α φ_μ φ_ν
        + 2 v_σ^{αα} ∇ρ_α·(∇φ_μ φ_ν + φ_μ ∇φ_ν)
        + v_σ^{αβ} ∇ρ_β·(∇φ_μ φ_ν + φ_μ ∇φ_ν) ]``

    Диагональный член несёт множитель 2, потому что ``σ_αα = ∇ρ_α·∇ρ_α`` и
    ``∂σ_αα/∂∇ρ_α = 2∇ρ_α``; кросс-член ``σ_αβ = ∇ρ_α·∇ρ_β`` зависит от
    ``∇ρ_α`` линейно, поэтому его множитель 1. В замкнутом пределе
    (ρ_α = ρ_β = ρ/2, ∇ρ_α = ∇ρ_β = ∇ρ/2) сумма свёртывается ровно в
    ``v_ρ φ_μ φ_ν + 2 v_σ ∇ρ·(∇φ_μ φ_ν + φ_μ ∇ρ_ν)`` неполяризованного
    функционала — проверка UKS==RKS на замкнутой оболочке опирается именно на
    это тождество.

    Энергия — ``Σ_g w_g ρ_общ ε_xc``, как в RKS: нелинейный функционал, и след
    ``D·V_xc`` здесь неверен.
    """
    density = np.stack([rho_alpha, rho_beta], axis=0)
    grad_density = np.stack([grad_alpha, grad_beta], axis=0)
    xc = functional.evaluate_spin(grid.points, density, grad_density)
    rho_total = rho_alpha + rho_beta
    energy = float(np.sum(grid.weights * rho_total * xc.energy_density))
    # LDA-функционал не зависит от градиентов: vsigma None ⇒ v_σ ≡ 0.
    v_sigma = xc.vsigma if xc.vsigma is not None else np.zeros((2, 2, rho_alpha.size), dtype=float)

    def channel(
        v_rho: np.ndarray,
        v_diag: np.ndarray,
        v_cross: np.ndarray,
        grad_self: np.ndarray,
        grad_other: np.ndarray,
    ) -> np.ndarray:
        scaled_rho = grid.weights * v_rho
        matrix = np.asarray(np.einsum("p,pg,ph->gh", scaled_rho, values, values, optimize=True))
        # Диагональный член: множитель 2 = ∂(∇ρ·∇ρ)/∂∇ρ.
        projection_self = np.einsum("pgd,pd->pg", gradients, grad_self, optimize=True)
        scaled_diag = 2.0 * grid.weights * v_diag
        matrix += np.einsum("p,pg,ph->gh", scaled_diag, projection_self, values, optimize=True)
        matrix += np.einsum("p,pg,ph->gh", scaled_diag, values, projection_self, optimize=True)
        # Кросс-член: множитель 1 (∂(∇ρ_α·∇ρ_β)/∂∇ρ_α = ∇ρ_β).
        projection_other = np.einsum("pgd,pd->pg", gradients, grad_other, optimize=True)
        scaled_cross = grid.weights * v_cross
        matrix += np.einsum("p,pg,ph->gh", scaled_cross, projection_other, values, optimize=True)
        matrix += np.einsum("p,pg,ph->gh", scaled_cross, values, projection_other, optimize=True)
        return matrix

    matrix_alpha = channel(xc.vrho[0], v_sigma[0, 0], v_sigma[0, 1], grad_alpha, grad_beta)
    matrix_beta = channel(xc.vrho[1], v_sigma[1, 1], v_sigma[0, 1], grad_beta, grad_alpha)
    return matrix_alpha, matrix_beta, energy


def run_rks(
    basis: BasisSet,
    molecule: Molecule,
    functional: ExchangeCorrelationFunctional,
    settings: ScfSettings | None = None,
    *,
    integrals: PrecomputedIntegrals | None = None,
    grid: QuadratureGrid | None = None,
    grid_preset: GridPreset = GridPreset.FINE,
) -> RksResult:
    """Выполняет RKS-расчёт в приближении LDA.

    Сетка и интегралы можно передать готовыми: тогда их стоимость относится к
    этапу вызывающей стороны, а не к ``scf``.
    """
    config = settings or ScfSettings()
    started = time.perf_counter()
    strategies: list[str] = ["core-hamiltonian-guess"]

    electrons = molecule.n_electrons
    if electrons % 2 != 0:
        msg = (
            f"RKS требует чётного числа электронов (замкнутая оболочка), получено {electrons}. "
            "Для открытой оболочки используйте run_uks (спиново-поляризованный UKS)."
        )
        raise ValueError(msg)
    n_occupied = electrons // 2

    prepared = integrals if integrals is not None else build_integrals(basis, molecule)
    quadrature = grid if grid is not None else build_grid(molecule, grid_preset)
    core = prepared.core
    eri = prepared.eri
    v_nuc = nuclear_repulsion(molecule)
    orthogonalizer = canonical_orthogonalizer(prepared.overlap)

    # Базисные функции и их градиенты в точках сетки не зависят от плотности,
    # поэтому вычисляются один раз на расчёт, а не на итерацию. Градиенты нужны
    # GGA; для LDA они вычисляются всё равно — отдельный путь без них означал бы
    # две версии одного и того же цикла.
    values, gradients = evaluate_basis_with_gradients(basis, molecule, quadrature.points)

    def xc_at(current_density: np.ndarray) -> tuple[np.ndarray, float]:
        """XC-матрица и энергия для текущей плотности."""
        rho = density_at_points(values, current_density)
        rho_gradient = density_gradient_at_points(values, gradients, current_density)
        return xc_matrix_and_energy(quadrature, values, gradients, rho, rho_gradient, functional)

    def diagonalize(fock_prime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symmetric = 0.5 * (fock_prime + fock_prime.T)
        energies, coefficients_prime = np.linalg.eigh(symmetric)
        return energies, orthogonalizer @ coefficients_prime, coefficients_prime

    alpha_exchange = functional.exact_exchange_fraction
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
    xc_energy = 0.0

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration
        v_xc, xc_energy = xc_at(density)
        coulomb = coulomb_matrix(density, eri)
        fock = core + coulomb + v_xc
        exact_exchange_energy = 0.0
        if alpha_exchange > 0.0:
            exchange = np.einsum("ls,ulvs->uv", density, eri, optimize=True)
            # E_x^exact = −¼α·D:K, значит ∂E/∂D = −½α·K: в фокиане коэффициент ½
            # при α — тот же, что у RHF-обмена, а не «просто α». Перепутать
            # легко, потому что в энергию входит ¼α, а не ½α, и оба числа
            # выглядят правдоподобно.
            fock = fock - 0.5 * alpha_exchange * exchange
            exact_exchange_energy = -0.25 * alpha_exchange * float(np.sum(density * exchange))

        # E = Σ D(H + ½J) + E_xc. Обменно-корреляционный потенциал входит в
        # фокиан, но в энергию — только E_xc: функционал нелинеен по плотности,
        # поэтому множителя ½, как у HF-обмена, здесь нет, а след D·V_xc брать
        # нельзя — он даёт совсем другую величину (для воды/STO-3G −11.69 вместо
        # −8.87) и удвоение даёт ошибку в десятки хартри.
        energy = (
            float(np.sum(density * (core + 0.5 * coulomb)))
            + xc_energy
            + exact_exchange_energy
            + v_nuc
        )
        energy_change = energy - previous_energy

        fock_prime = orthogonalizer.T @ fock @ orthogonalizer
        occupied_prime = coefficients_prime[:, :n_occupied]
        density_prime = 2.0 * occupied_prime @ occupied_prime.T
        commutator = density_prime @ fock_prime - fock_prime @ density_prime
        diis_error = float(np.max(np.abs(commutator)))

        strategy = "plain"
        effective_fock_prime = fock_prime

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

        if level_shift_engaged(history, diis_error, config, iteration):
            if "level-shift" not in strategies:
                strategies.append("level-shift")
            identity = np.eye(fock_prime.shape[0])
            effective_fock_prime = fock_prime + config.level_shift * (
                identity - occupied_prime @ occupied_prime.T
            )
            strategy = "level-shift"

        energies, coefficients, coefficients_prime = diagonalize(effective_fock_prime)
        new_density = density_from_coefficients(coefficients, n_occupied)

        regressed = iteration > 1 and energy_change > 0.0
        if iteration <= config.damping_rounds or regressed:
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

    v_xc, xc_energy = xc_at(density)
    coulomb = coulomb_matrix(density, eri)
    total = float(np.sum(density * (core + 0.5 * coulomb))) + xc_energy + v_nuc
    if alpha_exchange > 0.0:
        # Энергия пересобирается на сошедшейся плотности, поэтому обменный
        # член нужно добавить и здесь — иначе он есть в истории итераций, но
        # теряется в возвращаемом результате.
        exchange = np.einsum("ls,ulvs->uv", density, eri, optimize=True)
        total -= 0.25 * alpha_exchange * float(np.sum(density * exchange))
    return RksResult(
        total_energy=total,
        electronic_energy=total - v_nuc,
        nuclear_repulsion=v_nuc,
        xc_energy=xc_energy,
        exact_exchange_fraction=alpha_exchange,
        orbital_energies=tuple(float(value) for value in energies),
        coefficients=coefficients,
        density=density,
        converged=converged,
        iterations=iterations,
        grid_points=quadrature.n_points,
        v_xc=v_xc,
        history=history,
        strategies_used=tuple(strategies),
        elapsed_seconds=time.perf_counter() - started,
    )


def run_uks(
    basis: BasisSet,
    molecule: Molecule,
    functional: ExchangeCorrelationFunctional,
    settings: ScfSettings | None = None,
    *,
    integrals: PrecomputedIntegrals | None = None,
    grid: QuadratureGrid | None = None,
    grid_preset: GridPreset = GridPreset.FINE,
) -> UksResult:
    """Выполняет UKS-расчёт — DFT с разделёнными спиновыми каналами.

    Структура — как в UHF (два фокиана, кулон по полной плотности, обмен по
    своему каналу), плюс обменно-корреляционный член, как в RKS: потенциал
    ``V_xc^σ`` строится по спиново-разделённым ``v_ρ^σ`` и ``v_σ^{στ}``.

    Фокиан канала α:

    ``F^α = H + J + V_xc^α − ½α K_α``

    где ``α`` — доля точного обмена функционала (PBE0: 0.25, B3LYP: 0.20).
    В энергии обменная часть идёт с ¼α:

    ``E = Σ D_tot H + ½ Σ D_tot J + E_xc − ¼α Σ_σ D_σ:K_σ + V_ядер``

    При α = 0 расчёт — «чистый» GGA/LDA в спин-разделённом представлении; при
    замкнутой оболочке (n_α = n_β) решение обязано совпасть с RKS — это
    встроенная проверка корректности спинового раздела.

    Точность: сетка и функционалы — те же, что в RKS (см. их ограничения);
    спин-ядра сверены с LibXC 7.0.0 по 60 случайным точкам до ≤1e−14.
    """
    config = settings or ScfSettings()
    started = time.perf_counter()
    strategies: list[str] = ["core-hamiltonian-guess"]

    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)

    prepared = integrals if integrals is not None else build_integrals(basis, molecule)
    quadrature = grid if grid is not None else build_grid(molecule, grid_preset)
    core = prepared.core
    eri = prepared.eri
    v_nuc = nuclear_repulsion(molecule)
    orthogonalizer = canonical_orthogonalizer(prepared.overlap)

    values, gradients = evaluate_basis_with_gradients(basis, molecule, quadrature.points)

    def xc_at(d_alpha: np.ndarray, d_beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Спиновые XC-матрицы и энергия для текущих плотностей."""
        rho_a = density_at_points(values, d_alpha)
        rho_b = density_at_points(values, d_beta)
        grad_a = density_gradient_at_points(values, gradients, d_alpha)
        grad_b = density_gradient_at_points(values, gradients, d_beta)
        return xc_matrix_and_energy_spin(
            quadrature, values, gradients, rho_a, rho_b, grad_a, grad_b, functional
        )

    def diagonalize(fock_prime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symmetric = 0.5 * (fock_prime + fock_prime.T)
        energies, coefficients_prime = np.linalg.eigh(symmetric)
        return energies, orthogonalizer @ coefficients_prime, coefficients_prime

    alpha_exchange = functional.exact_exchange_fraction
    alpha_energies, alpha_coefficients, alpha_prime = diagonalize(
        orthogonalizer.T @ core @ orthogonalizer
    )
    beta_energies, beta_coefficients, beta_prime = (
        alpha_energies,
        alpha_coefficients,
        alpha_prime,
    )
    # Стартовая догадка — как в UHF: одинаковые α и β из ядра Гамильтона.
    # Для замкнутой оболочки это RKS-стартовое решение, для открытой —
    # Hund-совместимое (лишняя орбиталь занята в α).
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
    xc_energy = 0.0

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration
        v_xc_alpha, v_xc_beta, xc_energy = xc_at(density_alpha, density_beta)
        density_total = density_alpha + density_beta
        coulomb = coulomb_matrix(density_total, eri)
        fock_alpha = core + coulomb + v_xc_alpha
        fock_beta = core + coulomb + v_xc_beta
        exact_exchange_energy = 0.0
        if alpha_exchange > 0.0:
            exchange_alpha = exchange_matrix(density_alpha, eri)
            exchange_beta = exchange_matrix(density_beta, eri)
            # E_x^exact = −½α Σ_σ D_σ:K_σ ⇒ ∂E/∂D_σ = −α K_σ. Коэффициент 1 при
            # K_σ (а не ½, как в RHF): спин-орбиталь занята **один** раз, и
            # точный обмен UKS — это доля α от UHF-обмена −½Σ_σD_σ:K_σ. Ловушка:
            # скопировать ½/¼ из RKS — значит потерять фактор 2, невидимый по
            # сходимости, но сдвигающий энергию гибрида вдвое. В замкнутом
            # пределе (D_σ = D/2) −½αΣ_σD_σK_σ сворачивается ровно в −¼αD:K RKS.
            fock_alpha = fock_alpha - alpha_exchange * exchange_alpha
            fock_beta = fock_beta - alpha_exchange * exchange_beta
            exact_exchange_energy = (
                -0.5
                * alpha_exchange
                * (
                    float(np.sum(density_alpha * exchange_alpha))
                    + float(np.sum(density_beta * exchange_beta))
                )
            )

        # E = Σ D_tot H + ½ Σ D_tot J + E_xc − ½α Σ D_σ:K_σ + V_ядер.
        # XC-член — только E_xc (нелинейный функционал, след D·V_xc не равен
        # энергии) — то же, что в RKS; кулоновский член здесь с ½, как в RHF,
        # а не с 1 (это ловушка: Σ_σ D_σF_σ содержит J дважды).
        energy = (
            float(np.sum(density_total * core) + 0.5 * np.sum(density_total * coulomb))
            + xc_energy
            + exact_exchange_energy
            + v_nuc
        )
        energy_change = energy - previous_energy

        fock_alpha_prime = orthogonalizer.T @ fock_alpha @ orthogonalizer
        fock_beta_prime = orthogonalizer.T @ fock_beta @ orthogonalizer
        # Коммутатор — с орбиталями предыдущей диагонализации (как в UHF):
        # [D', F'], где D' строится из ортогонализированных занятых орбиталей.
        occupied_alpha = alpha_prime[:, :n_alpha]
        occupied_beta = beta_prime[:, :n_beta]
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

    v_xc_alpha, v_xc_beta, xc_energy = xc_at(density_alpha, density_beta)
    density_total = density_alpha + density_beta
    coulomb = coulomb_matrix(density_total, eri)
    fock_alpha = core + coulomb + v_xc_alpha
    fock_beta = core + coulomb + v_xc_beta
    exact_exchange_energy = 0.0
    if alpha_exchange > 0.0:
        exchange_alpha = exchange_matrix(density_alpha, eri)
        exchange_beta = exchange_matrix(density_beta, eri)
        fock_alpha = fock_alpha - alpha_exchange * exchange_alpha
        fock_beta = fock_beta - alpha_exchange * exchange_beta
        exact_exchange_energy = (
            -0.5
            * alpha_exchange
            * (
                float(np.sum(density_alpha * exchange_alpha))
                + float(np.sum(density_beta * exchange_beta))
            )
        )
    total = (
        float(np.sum(density_total * core) + 0.5 * np.sum(density_total * coulomb))
        + xc_energy
        + exact_exchange_energy
        + v_nuc
    )
    return UksResult(
        total_energy=total,
        electronic_energy=total - v_nuc,
        nuclear_repulsion=v_nuc,
        xc_energy=xc_energy,
        exact_exchange_fraction=alpha_exchange,
        alpha_energies=tuple(float(value) for value in alpha_energies),
        beta_energies=tuple(float(value) for value in beta_energies),
        alpha_coefficients=alpha_coefficients,
        beta_coefficients=beta_coefficients,
        density_alpha=density_alpha,
        density_beta=density_beta,
        s_squared=spin_contamination(
            alpha_coefficients, beta_coefficients, n_alpha, n_beta, prepared.overlap
        ),
        converged=converged,
        iterations=iterations,
        grid_points=quadrature.n_points,
        v_xc_alpha=v_xc_alpha,
        v_xc_beta=v_xc_beta,
        history=history,
        strategies_used=tuple(strategies),
        elapsed_seconds=time.perf_counter() - started,
    )


def run_uks(
    basis: BasisSet,
    molecule: Molecule,
    functional: ExchangeCorrelationFunctional,
    settings: ScfSettings | None = None,
    *,
    integrals: PrecomputedIntegrals | None = None,
    grid: QuadratureGrid | None = None,
    grid_preset: GridPreset = GridPreset.FINE,
) -> UksResult:
    """Выполняет UKS-расчёт — DFT с разделёнными спиновыми каналами.

    Структура — как в UHF (два фокиана, кулон по полной плотности, обмен по
    своему каналу), плюс обменно-корреляционный член, как в RKS: потенциал
    ``V_xc^σ`` строится по спиново-разделённым ``v_ρ^σ`` и ``v_σ^{στ}``.

    Фокиан канала α:

    ``F^α = H + J + V_xc^α − ½α K_α``

    где ``α`` — доля точного обмена функционала (PBE0: 0.25, B3LYP: 0.20).
    В энергии обменная часть идёт с ¼α:

    ``E = Σ D_tot H + ½ Σ D_tot J + E_xc − ¼α Σ_σ D_σ:K_σ + V_ядер``

    При α = 0 расчёт — «чистый» GGA/LDA в спин-разделённом представлении; при
    замкнутой оболочке (n_α = n_β) решение обязано совпасть с RKS — это
    встроенная проверка корректности спинового раздела.

    Точность: сетка и функционалы — те же, что в RKS (см. их ограничения);
    спин-ядра сверены с LibXC 7.0.0 по 60 случайным точкам до ≤1e−14.
    """
    config = settings or ScfSettings()
    started = time.perf_counter()
    strategies: list[str] = ["core-hamiltonian-guess"]

    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)

    prepared = integrals if integrals is not None else build_integrals(basis, molecule)
    quadrature = grid if grid is not None else build_grid(molecule, grid_preset)
    core = prepared.core
    eri = prepared.eri
    v_nuc = nuclear_repulsion(molecule)
    orthogonalizer = canonical_orthogonalizer(prepared.overlap)

    values, gradients = evaluate_basis_with_gradients(basis, molecule, quadrature.points)

    def xc_at(d_alpha: np.ndarray, d_beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Спиновые XC-матрицы и энергия для текущих плотностей."""
        rho_a = density_at_points(values, d_alpha)
        rho_b = density_at_points(values, d_beta)
        grad_a = density_gradient_at_points(values, gradients, d_alpha)
        grad_b = density_gradient_at_points(values, gradients, d_beta)
        return xc_matrix_and_energy_spin(
            quadrature, values, gradients, rho_a, rho_b, grad_a, grad_b, functional
        )

    def diagonalize(fock_prime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symmetric = 0.5 * (fock_prime + fock_prime.T)
        energies, coefficients_prime = np.linalg.eigh(symmetric)
        return energies, orthogonalizer @ coefficients_prime, coefficients_prime

    alpha_exchange = functional.exact_exchange_fraction
    alpha_energies, alpha_coefficients, alpha_prime = diagonalize(
        orthogonalizer.T @ core @ orthogonalizer
    )
    beta_energies, beta_coefficients, beta_prime = (
        alpha_energies,
        alpha_coefficients,
        alpha_prime,
    )
    # Стартовая догадка — как в UHF: одинаковые α и β из ядра Гамильтона.
    # Для замкнутой оболочки это RKS-стартовое решение, для открытой —
    # Hund-совместимое (лишняя орбиталь занята в α).
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
    xc_energy = 0.0

    for iteration in range(1, config.max_iterations + 1):
        iterations = iteration
        v_xc_alpha, v_xc_beta, xc_energy = xc_at(density_alpha, density_beta)
        density_total = density_alpha + density_beta
        coulomb = coulomb_matrix(density_total, eri)
        fock_alpha = core + coulomb + v_xc_alpha
        fock_beta = core + coulomb + v_xc_beta
        exact_exchange_energy = 0.0
        if alpha_exchange > 0.0:
            exchange_alpha = exchange_matrix(density_alpha, eri)
            exchange_beta = exchange_matrix(density_beta, eri)
            # E_x^exact = −½α Σ_σ D_σ:K_σ ⇒ ∂E/∂D_σ = −α K_σ. Коэффициент 1 при
            # K_σ (а не ½, как в RHF): спин-орбиталь занята **один** раз, и
            # точный обмен UKS — это доля α от UHF-обмена −½Σ_σD_σ:K_σ. Ловушка:
            # скопировать ½/¼ из RKS — значит потерять фактор 2, невидимый по
            # сходимости, но сдвигающий энергию гибрида вдвое. В замкнутом
            # пределе (D_σ = D/2) −½αΣ_σD_σK_σ сворачивается ровно в −¼αD:K RKS.
            fock_alpha = fock_alpha - alpha_exchange * exchange_alpha
            fock_beta = fock_beta - alpha_exchange * exchange_beta
            exact_exchange_energy = (
                -0.5
                * alpha_exchange
                * (
                    float(np.sum(density_alpha * exchange_alpha))
                    + float(np.sum(density_beta * exchange_beta))
                )
            )

        # E = Σ D_tot H + ½ Σ D_tot J + E_xc − ½α Σ D_σ:K_σ + V_ядер.
        # XC-член — только E_xc (нелинейный функционал, след D·V_xc не равен
        # энергии) — то же, что в RKS; кулоновский член здесь с ½, как в RHF,
        # а не с 1 (это ловушка: Σ_σ D_σF_σ содержит J дважды).
        energy = (
            float(np.sum(density_total * core) + 0.5 * np.sum(density_total * coulomb))
            + xc_energy
            + exact_exchange_energy
            + v_nuc
        )
        energy_change = energy - previous_energy

        fock_alpha_prime = orthogonalizer.T @ fock_alpha @ orthogonalizer
        fock_beta_prime = orthogonalizer.T @ fock_beta @ orthogonalizer
        # Коммутатор — с орбиталями предыдущей диагонализации (как в UHF):
        # [D', F'], где D' строится из ортогонализированных занятых орбиталей.
        occupied_alpha = alpha_prime[:, :n_alpha]
        occupied_beta = beta_prime[:, :n_beta]
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

    v_xc_alpha, v_xc_beta, xc_energy = xc_at(density_alpha, density_beta)
    density_total = density_alpha + density_beta
    coulomb = coulomb_matrix(density_total, eri)
    fock_alpha = core + coulomb + v_xc_alpha
    fock_beta = core + coulomb + v_xc_beta
    exact_exchange_energy = 0.0
    if alpha_exchange > 0.0:
        exchange_alpha = exchange_matrix(density_alpha, eri)
        exchange_beta = exchange_matrix(density_beta, eri)
        fock_alpha = fock_alpha - alpha_exchange * exchange_alpha
        fock_beta = fock_beta - alpha_exchange * exchange_beta
        exact_exchange_energy = (
            -0.5
            * alpha_exchange
            * (
                float(np.sum(density_alpha * exchange_alpha))
                + float(np.sum(density_beta * exchange_beta))
            )
        )
    total = (
        float(np.sum(density_total * core) + 0.5 * np.sum(density_total * coulomb))
        + xc_energy
        + exact_exchange_energy
        + v_nuc
    )
    return UksResult(
        total_energy=total,
        electronic_energy=total - v_nuc,
        nuclear_repulsion=v_nuc,
        xc_energy=xc_energy,
        exact_exchange_fraction=alpha_exchange,
        alpha_energies=tuple(float(value) for value in alpha_energies),
        beta_energies=tuple(float(value) for value in beta_energies),
        alpha_coefficients=alpha_coefficients,
        beta_coefficients=beta_coefficients,
        density_alpha=density_alpha,
        density_beta=density_beta,
        s_squared=spin_contamination(
            alpha_coefficients, beta_coefficients, n_alpha, n_beta, prepared.overlap
        ),
        converged=converged,
        iterations=iterations,
        grid_points=quadrature.n_points,
        v_xc_alpha=v_xc_alpha,
        v_xc_beta=v_xc_beta,
        history=history,
        strategies_used=tuple(strategies),
        elapsed_seconds=time.perf_counter() - started,
    )


def level_shift_engaged(
    history: list[ScfHistory], diis_error: float, config: ScfSettings, iteration: int
) -> bool:
    """Включать ли сдвиг уровней на этой итерации.

    Условие то же, что в RHF: невязка перестала падать за окно DIIS и всё ещё
    выше допуска. Вынесено в функцию, чтобы правило не разъезжалось между
    решателями — расхождение означало бы разную сходимость одного расчёта.
    """
    del iteration
    return (
        len(history) >= config.diis_space + config.diis_start
        and diis_error > config.stall_ratio * history[-config.diis_space].diis_error
        and diis_error > config.density_tolerance
    )
