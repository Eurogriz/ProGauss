"""Meta-GGA: TPSS-x, PBC-корреляция и гибрид TPSSh (RKS, энергия в точке).

Три линии проверки, как у остальных функционалов:

* **ядра** сверяются с LibXC (через PySCF) по энергии и всем трём потенциалам
  ``(v_ρ, v_σ, v_τ)`` — ошибка формулы или цепного правила;
* **энергия RKS** сверяется с ``pyscf.dft.RKS`` — ошибка решателя (член
  ``v_τ`` в фокиане, доля точного обмена 0.10, кинетическая плотность);
* **отказы**: UKS, аналитический градиент, оптимизация и частоты для
  meta-GGA не реализованы и должны отклоняться явно, а не приближаться (§54 ТЗ).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    GridPreset,
    MethodSpec,
    OptimizationSpec,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.capabilities import Availability
from quantumlab.engine.constants import angstrom_to_bohr
from quantumlab.engine.contracts import EngineRequest, ExchangeCorrelationFunctional
from quantumlab.engine.dft import run_rks, run_uks
from quantumlab.engine.functional import (
    FUNCTIONALS,
    Pbe,
    TpssCorrelation,
    TpssExchange,
    Tpssh,
    density_at_points,
    density_gradient_at_points,
    evaluate_basis_with_gradients,
    get_functional,
    kinetic_density_at_points,
)
from quantumlab.engine.gradients import rks_gradient
from quantumlab.engine.quadrature import build_grid
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.registry import default_registry
from quantumlab.engine.scf import ScfSettings, run_rhf
from quantumlab.errors import CombinationUnavailableError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)


def _physical_points(size: int = 300, seed: int = 7) -> tuple[np.ndarray, ...]:
    """Точки в физической области ``σ ≤ 8ρτ`` (условие Фермиевской дырки)."""
    generator = np.random.default_rng(seed)
    rho = 10 ** generator.uniform(-3.0, 1.5, size)
    tau_thomas_fermi = 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    tau = tau_thomas_fermi * generator.uniform(0.5, 3.0, size)
    sigma = generator.uniform(0.01, 0.9, size) * 8.0 * rho * tau
    gradient = np.zeros((size, 3))
    gradient[:, 0] = np.sqrt(sigma)
    return rho, sigma, tau, gradient


def test_tpssh_declares_ten_percent_exact_exchange() -> None:
    """TPSSh — гибрид meta-GGA с долей точного обмена 0.10 (TPSS0 — это 0.25)."""
    functional = Tpssh()
    assert functional.is_hybrid
    assert functional.functional_class == "mgga"
    assert functional.requires_tau
    assert functional.exact_exchange_fraction == pytest.approx(0.10)
    assert functional.dft_exchange_fraction == pytest.approx(0.90)
    assert functional.exact_exchange_fraction + functional.dft_exchange_fraction == 1.0


def test_tpssh_conforms_to_the_functional_protocol() -> None:
    assert isinstance(get_functional("tpssh"), ExchangeCorrelationFunctional)
    assert FUNCTIONALS["tpssh"] is Tpssh
    assert not Pbe().requires_tau


def test_tpssh_is_weighted_sum_of_its_parts() -> None:
    rho, _sigma, tau, gradient = _physical_points(50)
    points = np.zeros((rho.size, 3))
    whole = Tpssh().evaluate(points, rho, gradient, tau=tau)
    exchange = TpssExchange().evaluate(points, rho, gradient, tau=tau)
    correlation = TpssCorrelation().evaluate(points, rho, gradient, tau=tau)
    assert np.allclose(
        whole.energy_density, 0.9 * exchange.energy_density + correlation.energy_density
    )
    assert whole.vtau is not None and exchange.vtau is not None and correlation.vtau is not None
    assert np.allclose(whole.vtau, 0.9 * exchange.vtau + correlation.vtau)


@pytest.mark.parametrize("functional", [TpssExchange(), TpssCorrelation(), Tpssh()])
def test_meta_gga_requires_tau_and_gradient(functional: ExchangeCorrelationFunctional) -> None:
    rho, _sigma, _tau, gradient = _physical_points(5)
    points = np.zeros((5, 3))
    with pytest.raises(ValueError, match="tau"):
        functional.evaluate(points, rho, gradient)
    with pytest.raises(ValueError, match="градиент"):
        functional.evaluate(points, rho, None, tau=rho)


@pytest.mark.parametrize("functional", [TpssExchange(), TpssCorrelation(), Tpssh()])
def test_meta_gga_spin_variant_is_refused(functional: ExchangeCorrelationFunctional) -> None:
    with pytest.raises(NotImplementedError, match="meta-GGA"):
        functional.evaluate_spin(np.zeros((1, 3)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="evaluate_spin"):
        functional.evaluate(
            np.zeros((1, 3)), np.ones(1), np.zeros((1, 3)), spin_polarized=True, tau=np.ones(1)
        )


@pytest.mark.parametrize(
    ("functional", "libxc_name"),
    [(TpssExchange(), "MGGA_X_TPSS"), (TpssCorrelation(), "MGGA_C_TPSS")],
)
def test_kernels_match_libxc(functional: ExchangeCorrelationFunctional, libxc_name: str) -> None:
    libxc = pytest.importorskip("pyscf.dft.libxc", reason="LibXC нужен для независимой сверки")
    rho, _sigma, tau, gradient = _physical_points()
    ours = functional.evaluate(np.zeros((rho.size, 3)), rho, gradient, tau=tau)
    theirs_e, theirs_v = libxc.eval_xc(
        libxc_name,
        np.array([rho, gradient[:, 0], gradient[:, 1], gradient[:, 2], np.zeros_like(rho), tau]),
        spin=0,
        deriv=1,
    )[:2]
    assert ours.vsigma is not None and ours.vtau is not None
    assert np.allclose(ours.energy_density, theirs_e, rtol=1e-8, atol=1e-14)
    assert np.allclose(ours.vrho, theirs_v[0], rtol=1e-8, atol=1e-12)
    assert np.allclose(ours.vsigma, theirs_v[1], rtol=1e-8, atol=1e-12)
    assert np.allclose(ours.vtau, theirs_v[3], rtol=1e-6, atol=1e-9)


def test_kernels_potentials_are_derivatives_of_energy() -> None:
    """``v_ρ, v_σ, v_τ`` — центральные разности ``ρ ε`` (независимо от LibXC)."""
    rho, sigma, tau, _gradient = _physical_points(40, seed=3)

    def volume_energy(r: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        gradient = np.zeros((r.size, 3))
        gradient[:, 0] = np.sqrt(s)
        out = Tpssh().evaluate(np.zeros((r.size, 3)), r, gradient, tau=t)
        return np.asarray(r * out.energy_density)

    base = Tpssh().evaluate(
        np.zeros((rho.size, 3)), rho, np.stack([np.sqrt(sigma), 0 * rho, 0 * rho], axis=1), tau=tau
    )
    assert base.vsigma is not None and base.vtau is not None
    for variable, analytic in ((0, base.vrho), (1, base.vsigma), (2, base.vtau)):
        args = [rho.copy(), sigma.copy(), tau.copy()]
        step = 1e-6 * args[variable]
        plus = [a.copy() for a in args]
        minus = [a.copy() for a in args]
        plus[variable] = plus[variable] + step
        minus[variable] = minus[variable] - step
        numeric = (volume_energy(*plus) - volume_energy(*minus)) / (2.0 * step)
        assert np.allclose(analytic, numeric, rtol=2e-5, atol=1e-9), variable


def test_kinetic_density_matches_orbital_definition(water: Molecule) -> None:
    """``τ = ½ Σ_i n_i |∇ψ_i|²`` и его интеграл положителен и вещественен."""
    basis = build_basis("sto-3g", water)
    grid = build_grid(water, GridPreset.FINE)
    _values, gradients = evaluate_basis_with_gradients(basis, water, grid.points)
    scf = run_rhf(basis, water)
    tau = kinetic_density_at_points(gradients, scf.density)
    occupied = scf.coefficients[:, :5]
    orbital_gradients = np.einsum("pgd,gi->pid", gradients, occupied)
    reference = 0.5 * 2.0 * np.sum(orbital_gradients**2, axis=(1, 2))
    assert np.allclose(tau, reference, atol=1e-12)
    assert np.all(tau >= -1e-14)
    assert float(np.sum(grid.weights * tau)) > 0.0


def test_density_and_gradient_consistent_with_tau_sigma_bound(water: Molecule) -> None:
    """В молекуле условие ``σ ≤ 8ρτ`` выполняется почти на всех точках сетки."""
    basis = build_basis("sto-3g", water)
    grid = build_grid(water, GridPreset.FINE)
    values, gradients = evaluate_basis_with_gradients(basis, water, grid.points)
    density = run_rhf(basis, water).density
    rho = density_at_points(values, density)
    sigma = np.sum(density_gradient_at_points(values, gradients, density) ** 2, axis=1)
    tau = kinetic_density_at_points(gradients, density)
    significant = rho > 1e-8
    assert (
        np.mean(sigma[significant] <= 8.0 * rho[significant] * tau[significant] * (1 + 1e-9)) > 0.99
    )


def test_rks_tpssh_energy_matches_pyscf(water: Molecule) -> None:
    pyscf = pytest.importorskip("pyscf", reason="PySCF нужен только для независимой сверки")
    pyscf_dft = pytest.importorskip("pyscf.dft", reason="PySCF DFT нужен для независимой сверки")
    basis = build_basis("sto-3g", water)
    ours = run_rks(basis, water, Tpssh(), grid_preset=GridPreset.ULTRAFINE)
    assert ours.converged
    assert ours.exact_exchange_fraction == pytest.approx(0.10)

    theirs = pyscf.gto.M(
        atom=[(atom.symbol, atom.position) for atom in water.atoms],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    their_scf = pyscf_dft.RKS(theirs)
    their_scf.xc = "TPSSH"
    their_scf.grids.atom_grid = (120, 974)
    their_scf.verbose = 0
    their_scf.run(conv_tol=1e-12)
    assert ours.total_energy == pytest.approx(float(their_scf.e_tot), abs=5e-6)


def test_engine_runs_tpssh_single_point(water: Molecule) -> None:
    spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=TheoryFamily.DFT, basis="sto-3g", functional="tpssh"),
    )
    result = ReferenceEngine().run(
        EngineRequest(job_id="mgga", spec=spec, molecule=water, threads=1)
    )
    assert result.energy_hartree < -75.0
    assert result.converged


def test_rks_tpssh_gradient_matches_pyscf(water: Molecule) -> None:
    """Аналитический градиент TPSSh (с членом ``v_τ ∂τ/∂R``) против PySCF.

    ``grid_response=False`` — та же модель, что у нас: сетка неподвижна в пространстве.
    """
    pyscf = pytest.importorskip("pyscf", reason="PySCF нужен только для независимой сверки")
    pyscf_dft = pytest.importorskip("pyscf.dft", reason="PySCF DFT нужен для независимой сверки")
    basis = build_basis("sto-3g", water)
    functional = Tpssh()
    grid = build_grid(water, GridPreset.ULTRAFINE)
    result = run_rks(basis, water, functional, grid=grid)
    ours = rks_gradient(basis, water, result, grid, functional).gradient

    theirs = pyscf.gto.M(
        atom=[(atom.symbol, atom.position) for atom in water.atoms],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    their_scf = pyscf_dft.RKS(theirs)
    their_scf.xc = "TPSSH"
    their_scf.grids.atom_grid = (120, 974)
    their_scf.conv_tol = 1e-12
    their_scf.run()
    method = their_scf.nuc_grad_method()
    method.grid_response = False
    reference = np.asarray(method.kernel())
    assert float(np.max(np.abs(ours - reference))) < 5e-6


def test_rks_tpssh_gradient_matches_finite_differences(water: Molecule) -> None:
    """Градиент согласован с производной собственной энергии на той же неподвижной сетке."""
    basis = build_basis("sto-3g", water)
    functional = Tpssh()
    grid = build_grid(water, GridPreset.FINE)
    result = run_rks(basis, water, functional, TIGHT, grid=grid)
    analytic = rks_gradient(basis, water, result, grid, functional).gradient

    step = 1e-3  # Å
    atom, axis = 1, 1  # H, ось y: вдоль связи, ненулевая компонента
    energies = []
    for sign in (+1, -1):
        atoms = list(water.atoms)
        position = list(atoms[atom].position)
        position[axis] += sign * step
        atoms[atom] = Atom(symbol=atoms[atom].symbol, position=tuple(position))  # type: ignore[arg-type]
        moved = Molecule(name="moved", atoms=tuple(atoms))
        # Сетка строится по исходной геометрии: энергия при неподвижной сетке —
        # ровно та величина, производную которой считает аналитический градиент.
        energies.append(
            run_rks(build_basis("sto-3g", moved), moved, functional, TIGHT, grid=grid).total_energy
        )
    numeric = (energies[0] - energies[1]) / (2 * step * angstrom_to_bohr(1.0))
    assert abs(analytic[atom, axis] - numeric) < 2e-5


def test_engine_optimizes_tpssh_with_the_analytic_gradient(water: Molecule) -> None:
    spec = CalculationSpec(
        task=Task.OPTIMIZATION,
        method=MethodSpec(theory=TheoryFamily.DFT, basis="sto-3g", functional="tpssh"),
        optimization=OptimizationSpec(max_steps=3, coordinates="cartesian"),
    )
    result = ReferenceEngine().run(
        EngineRequest(job_id="mgga", spec=spec, molecule=water, threads=1)
    )
    assert result.energy_hartree < -75.0
    assert result.final_molecule is not None


def test_engine_refuses_tpssh_for_open_shell() -> None:
    radical = Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), name="ch", multiplicity=2
    )
    spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=TheoryFamily.DFT, basis="sto-3g", functional="tpssh"),
    )
    with pytest.raises(CombinationUnavailableError):
        ReferenceEngine().run(EngineRequest(job_id="mgga", spec=spec, molecule=radical, threads=1))


def test_solvers_refuse_meta_gga_where_it_is_not_implemented(water: Molecule) -> None:
    basis = build_basis("sto-3g", water)
    with pytest.raises(NotImplementedError, match="UKS"):
        run_uks(basis, water, Tpssh())


def test_registry_reports_tpssh_as_partial_rks_only() -> None:
    registry = default_registry()
    assert registry.availability("functional:tpssh") is Availability.PARTIAL
    capability = registry.get("functional:tpssh")
    text = " ".join(capability.limitations)
    assert "RKS" in text and "meta-GGA" in text
    assert not registry.is_available("functional:m06")
