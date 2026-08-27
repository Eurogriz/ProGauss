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
from quantumlab.engine.functional import density_at_points, evaluate_basis
from quantumlab.engine.quadrature import QuadratureGrid, build_grid
from quantumlab.engine.scf import (
    PrecomputedIntegrals,
    ScfHistory,
    ScfResult,
    ScfSettings,
    _diis_extrapolate,
    build_integrals,
    canonical_orthogonalizer,
    coulomb_matrix,
    density_from_coefficients,
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


def xc_matrix_and_energy(
    values: np.ndarray,
    grid: QuadratureGrid,
    density: np.ndarray,
    functional: ExchangeCorrelationFunctional,
) -> tuple[np.ndarray, float]:
    """Обменно-корреляционная матрица и энергия на текущей плотности.

    Возвращает ``(V_xc, E_xc)``. Энергия считается как ``Σ_g w_g ρ_g ε_xc(ρ_g)``,
    а не через след ``D·V_xc``: для нелинейного функционала это разные величины,
    и верна именно первая.
    """
    rho = density_at_points(values, density)
    exc, vxc = functional.evaluate(grid.points, rho)
    energy = float(np.sum(grid.weights * rho * exc))
    scaled = grid.weights * vxc
    matrix = np.einsum("p,pg,ph->gh", scaled, values, values, optimize=True)
    return np.asarray(matrix), energy


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
            "Для открытой оболочки нужен UKS — он не реализован."
        )
        raise ValueError(msg)
    n_occupied = electrons // 2

    prepared = integrals if integrals is not None else build_integrals(basis, molecule)
    quadrature = grid if grid is not None else build_grid(molecule, grid_preset)
    core = prepared.core
    eri = prepared.eri
    v_nuc = nuclear_repulsion(molecule)
    orthogonalizer = canonical_orthogonalizer(prepared.overlap)

    # Базисные функции в точках сетки не зависят от плотности, поэтому
    # вычисляются один раз на расчёт, а не на итерацию.
    values = evaluate_basis(basis, molecule, quadrature.points)

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
        v_xc, xc_energy = xc_matrix_and_energy(values, quadrature, density, functional)
        coulomb = coulomb_matrix(density, eri)
        fock = core + coulomb + v_xc
        if alpha_exchange > 0.0:
            exchange = np.einsum("ls,ulvs->uv", density, eri, optimize=True)
            fock = fock - alpha_exchange * exchange

        # E = Σ D(H + ½J) + E_xc. Обменно-корреляционный потенциал входит в
        # фокиан, но в энергию — только E_xc: функционал нелинеен по плотности,
        # поэтому множителя ½, как у HF-обмена, здесь нет, а след D·V_xc брать
        # нельзя — он даёт совсем другую величину (для воды/STO-3G −11.69 вместо
        # −8.87) и удвоение даёт ошибку в десятки хартри.
        energy = float(np.sum(density * (core + 0.5 * coulomb))) + xc_energy + v_nuc
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

    v_xc, xc_energy = xc_matrix_and_energy(values, quadrature, density, functional)
    coulomb = coulomb_matrix(density, eri)
    total = float(np.sum(density * (core + 0.5 * coulomb))) + xc_energy + v_nuc
    return RksResult(
        total_energy=total,
        electronic_energy=total - v_nuc,
        nuclear_repulsion=v_nuc,
        xc_energy=xc_energy,
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
