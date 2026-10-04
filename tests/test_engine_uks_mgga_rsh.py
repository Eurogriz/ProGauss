"""Спин-поляризованный meta-GGA, M06/M06-2X и гибриды с разделением диапазона (ωB97X, ωB97X-D).

Каждая новая возможность сверяется с независимым оракулом — LibXC и PySCF:

* **ядра** (энергия и все потенциалы) — с LibXC через ``pyscf.dft.libxc``;
* **E и V на фиксированной плотности** — с ``pyscf.dft.numint.nr_uks`` (проверяет
  интегрирование на сетке и член ``v_τ`` фокиана отдельно от SCF);
* **SCF и аналитический градиент** — с PySCF на **той же сетке** (координаты и
  веса подставляются в ``mf.grids``) — иначе разница в сетках перекрыла бы
  разницу в формулах. PySCF стартует с нашей плотности: у открытой оболочки
  разные стартовые догадки приводят к разным стационарным точкам, а не к
  разным ответам на один вопрос;
* **интегралы** ``erf(ωr)/r`` — с ``mol.with_range_coulomb``;
* **устойчивость** meta-GGA — со вторым дифференциалом PySCF (``newton_ah``).

D2-CHG (дисперсия ωB97X-D) независимого программного оракула не имеет; она
проверяется аналитически и конечными разностями.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    DispersionCorrection,
    GridPreset,
    GridSpec,
    MethodSpec,
    ScfSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine import integrals as integral_module
from quantumlab.engine import stability as stab
from quantumlab.engine.basis import build_basis
from quantumlab.engine.contracts import EngineRequest, range_separation
from quantumlab.engine.dft import run_rks, run_uks, xc_matrix_and_energy_spin
from quantumlab.engine.dispersion_d2 import (
    chg_elements,
    dftd2_chg_contribution,
)
from quantumlab.engine.fock import SpinFockBuilder
from quantumlab.engine.functional import (
    density_at_points,
    density_gradient_at_points,
    evaluate_basis_with_gradients,
    get_functional,
    kinetic_density_at_points,
)
from quantumlab.engine.gradients import rks_gradient, uks_gradient
from quantumlab.engine.quadrature import build_grid
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import ExactExchange, ScfSettings, build_integrals
from quantumlab.engine.xc_meta import K_FACTOR_C
from quantumlab.engine.xc_wb97 import attenuation_erf
from quantumlab.errors import CombinationUnavailableError

pytestmark = pytest.mark.scientific

FIXTURES = Path(__file__).parent / "fixtures"
TIGHT = ScfSettings(energy_tolerance=1e-10, density_tolerance=1e-8, max_iterations=200)

#: Имя функционала в LibXC/PySCF. Для ωB97X-D берётся «чистый» функционал LibXC:
#: ``wb97x-d`` без суффикса PySCF трактует как «с дисперсией» и отказывается.
_PYSCF_NAME = {
    "tpssh": "TPSSH",
    "m06": "M06",
    "m062x": "M062X",
    "wb97x": "wb97x",
    "wb97x-d": "HYB_GGA_XC_WB97X_D",
}


def _water(multiplicity: int = 1) -> Molecule:
    return Molecule.from_xyz(
        (FIXTURES / "water.xyz").read_text(encoding="utf-8"),
        name="water",
        multiplicity=multiplicity,
    )


def _displaced(molecule: Molecule, index: int, axis: int, delta_angstrom: float) -> Molecule:
    """Копия молекулы со сдвигом одного атома вдоль одной оси, Å."""
    positions = [list(atom.position) for atom in molecule.atoms]
    positions[index][axis] += delta_angstrom
    return Molecule.from_atoms(
        [atom.symbol for atom in molecule.atoms],
        positions,
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )


def _pyscf_molecule(molecule: Molecule, spin: int) -> object:
    gto = pytest.importorskip("pyscf.gto")
    atom = "; ".join(
        f"{a.symbol} {a.position[0]:.10f} {a.position[1]:.10f} {a.position[2]:.10f}"
        for a in molecule.atoms
    )
    return gto.M(atom=atom, basis="sto-3g", spin=spin, unit="Angstrom", cart=True, verbose=0)


def _pyscf_ks(molecule: Molecule, spin: int, functional: str, grid: object) -> object:
    dft = pytest.importorskip("pyscf.dft")
    mol = _pyscf_molecule(molecule, spin)
    method = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    method.xc = _PYSCF_NAME[functional]
    method.grids.coords = grid.points  # type: ignore[attr-defined]
    method.grids.weights = grid.weights  # type: ignore[attr-defined]
    method.conv_tol = 1e-12
    return method


# --------------------------------------------------------------------------- #
# Ядра против LibXC
# --------------------------------------------------------------------------- #
def _random_points(n: int, *, tau: bool) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(11)
    rho_a = rng.uniform(0.01, 2.0, n)
    rho_b = rng.uniform(0.01, 2.0, n)
    if tau:
        tau_a = K_FACTOR_C * rho_a ** (5 / 3) * rng.uniform(0.3, 3.0, n)
        tau_b = K_FACTOR_C * rho_b ** (5 / 3) * rng.uniform(0.3, 3.0, n)
        s_aa = 8.0 * rho_a * tau_a * rng.uniform(0.0, 0.9, n)
        s_bb = 8.0 * rho_b * tau_b * rng.uniform(0.0, 0.9, n)
    else:
        tau_a = tau_b = np.zeros(n)
        s_aa = rho_a ** (8 / 3) * rng.uniform(0.01, 4.0, n)
        s_bb = rho_b ** (8 / 3) * rng.uniform(0.01, 4.0, n)
    cosine = rng.uniform(-1.0, 1.0, n)
    s_ab = cosine * np.sqrt(s_aa * s_bb)
    sine = np.sqrt(1.0 - cosine**2)
    grad_a = np.stack([np.sqrt(s_aa), 0 * s_aa, 0 * s_aa], axis=1)
    grad_b = np.stack(
        [np.sqrt(s_bb) * cosine, np.sqrt(s_bb) * sine, 0 * s_bb],
        axis=1,
    )
    del s_ab
    return rho_a, rho_b, grad_a, grad_b, tau_a, tau_b


@pytest.mark.parametrize("name", ["tpssh", "m06", "m062x", "wb97x", "wb97x-d"])
def test_spin_kernel_matches_libxc(name: str) -> None:
    libxc = pytest.importorskip("pyscf.dft.libxc")
    functional = get_functional(name)
    n = 300
    rho_a, rho_b, grad_a, grad_b, tau_a, tau_b = _random_points(n, tau=functional.requires_tau)
    points = np.zeros((n, 3))
    density = np.stack([rho_a, rho_b])
    gradient = np.stack([grad_a, grad_b])
    if functional.requires_tau:
        ours = functional.evaluate_spin_tau(  # type: ignore[union-attr]
            points, density, gradient, np.stack([tau_a, tau_b])
        )
    else:
        ours = functional.evaluate_spin(points, density, gradient)
    if functional.requires_tau:
        lib_a = np.vstack([rho_a, grad_a.T, 0 * rho_a, tau_a])
        lib_b = np.vstack([rho_b, grad_b.T, 0 * rho_b, tau_b])
    else:
        lib_a = np.vstack([rho_a, grad_a.T])
        lib_b = np.vstack([rho_b, grad_b.T])
    spec = _PYSCF_NAME[name]
    exc, vxc, _, _ = libxc.eval_xc(spec, (lib_a, lib_b), spin=1, deriv=1)
    reference = (rho_a + rho_b) * exc
    energy = ours.energy_density * (rho_a + rho_b)
    assert np.max(np.abs(energy - reference) / (1e-12 + np.abs(reference))) < 1e-8
    assert np.max(np.abs(ours.vrho[0] - vxc[0][:, 0]) / (1e-8 + np.abs(vxc[0][:, 0]))) < 1e-6
    assert np.max(np.abs(ours.vrho[1] - vxc[0][:, 1]) / (1e-8 + np.abs(vxc[0][:, 1]))) < 1e-6
    assert ours.vsigma is not None
    for ours_part, index in (
        (ours.vsigma[0, 0], 0),
        (ours.vsigma[0, 1], 1),
        (ours.vsigma[1, 1], 2),
    ):
        scale = 1e-8 + np.abs(vxc[1][:, index])
        assert np.max(np.abs(ours_part - vxc[1][:, index]) / scale) < 1e-6
    if functional.requires_tau:
        assert ours.vtau is not None
        for channel in (0, 1):
            scale = 1e-8 + np.abs(vxc[3][:, channel])
            assert np.max(np.abs(ours.vtau[channel] - vxc[3][:, channel]) / scale) < 1e-6


def test_closed_shell_form_equals_the_spin_form_at_zero_polarisation() -> None:
    """RKS-форма ядра на AD — это спиновая форма при ρ_α = ρ_β: проверка свёртки производных."""
    functional = get_functional("wb97x")
    n = 50
    rng = np.random.default_rng(5)
    rho = rng.uniform(0.05, 1.5, n)
    gradient = rng.normal(size=(n, 3)) * rho[:, None] ** (4 / 3)
    closed = functional.evaluate(np.zeros((n, 3)), rho, gradient)
    spin = functional.evaluate_spin(
        np.zeros((n, 3)), np.stack([rho / 2, rho / 2]), np.stack([gradient / 2, gradient / 2])
    )
    assert np.allclose(closed.energy_density, spin.energy_density, atol=1e-13)
    assert np.allclose(closed.vrho, spin.vrho[0], atol=1e-12)
    assert np.allclose(closed.vrho, spin.vrho[1], atol=1e-12)


def test_attenuation_function_limits_and_branch_continuity() -> None:
    """``F(a)``: ``F(0) = 1``, ``F ≈ 1/(36a²)`` при больших ``a``, ряд и формула сшиты."""
    from quantumlab.engine.xc_ad import Dual

    def value(a: float) -> float:
        return float(attenuation_erf(Dual(np.array([a]))).v[0])

    assert value(1e-4) == pytest.approx(1.0 - 8.0 / 3.0 * 1e-4 * math.sqrt(math.pi), abs=1e-7)
    assert value(1e3) == pytest.approx(1.0 / (36.0 * 1e6), rel=1e-6)
    below, above = value(1.35 - 1e-12), value(1.35 + 1e-12)
    assert abs(below - above) < 1e-12
    # Монотонное убывание: короткодействующий обмен ослабляется сильнее при больших a.
    grid = [value(a) for a in np.linspace(0.05, 6.0, 80)]
    assert all(x > y for x, y in pairwise(grid))


# --------------------------------------------------------------------------- #
# E и V на фиксированной плотности против nr_uks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["tpssh", "m06", "m062x", "wb97x"])
def test_uks_xc_matrix_and_energy_match_pyscf_on_a_fixed_density(name: str) -> None:
    scf = pytest.importorskip("pyscf.scf")
    dft = pytest.importorskip("pyscf.dft")
    molecule = Molecule.from_atoms(["C", "H"], [(0, 0, 0), (0, 0, 1.117)], multiplicity=2)
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule, GridPreset.COARSE)
    mol = _pyscf_molecule(molecule, 1)
    uhf = scf.UHF(mol).run()
    dm_a, dm_b = uhf.make_rdm1()
    values, gradients = evaluate_basis_with_gradients(basis, molecule, grid.points)
    functional = get_functional(name)
    tau_a = tau_b = None
    if functional.requires_tau:
        tau_a = kinetic_density_at_points(gradients, dm_a)
        tau_b = kinetic_density_at_points(gradients, dm_b)
    v_a, v_b, energy = xc_matrix_and_energy_spin(
        grid,
        values,
        gradients,
        density_at_points(values, dm_a),
        density_at_points(values, dm_b),
        density_gradient_at_points(values, gradients, dm_a),
        density_gradient_at_points(values, gradients, dm_b),
        functional,
        tau_a,
        tau_b,
    )
    pyscf_grid = dft.Grids(mol)
    pyscf_grid.coords = grid.points
    pyscf_grid.weights = grid.weights
    _, reference_energy, reference_v = dft.numint.NumInt().nr_uks(
        mol, pyscf_grid, _PYSCF_NAME[name], (dm_a, dm_b)
    )
    assert energy == pytest.approx(reference_energy, abs=1e-6)
    assert np.max(np.abs(v_a - reference_v[0])) < 1e-6
    assert np.max(np.abs(v_b - reference_v[1])) < 1e-6


# --------------------------------------------------------------------------- #
# Интегралы erf(ωr)/r
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("omega", [0.2, 0.3, 0.8])
def test_long_range_integrals_match_pyscf(omega: float) -> None:
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    ours = integral_module.build_electron_repulsion(basis, molecule, omega=omega)
    mol = _pyscf_molecule(molecule, 0)
    with mol.with_range_coulomb(omega):  # type: ignore[attr-defined]
        reference = mol.intor("int2e")  # type: ignore[attr-defined]
    # Базис STO-3G в PySCF и здесь отличается разрядами экспонент: невязка кулоновского
    # тензора без ослабления — тот же порядок, поэтому порог общий, а не подогнан.
    plain = integral_module.build_electron_repulsion(basis, molecule)
    plain_reference = mol.intor("int2e")  # type: ignore[attr-defined]
    baseline = float(np.max(np.abs(plain - plain_reference)))
    assert float(np.max(np.abs(ours - reference))) < max(10 * baseline, 1e-6)


def test_long_range_integrals_have_eightfold_symmetry_and_limits() -> None:
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    long_range = integral_module.build_electron_repulsion(basis, molecule, omega=0.3)
    for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert np.allclose(long_range, long_range.transpose(permutation), atol=1e-13)
    plain = integral_module.build_electron_repulsion(basis, molecule)
    # ω → ∞: erf(ωr)/r → 1/r.
    # Остаток ~π/ω²·ρ(0): при ω = 1e5 порядка 6e-9.
    huge = integral_module.build_electron_repulsion(basis, molecule, omega=1e5)
    assert np.max(np.abs(huge - plain)) < 3e-8
    # Для любого ω ослабленное ядро не больше кулоновского на диагонали (μν|μν) ≥ 0.
    diagonal = np.einsum("ijij->ij", long_range)
    assert np.all(diagonal <= np.einsum("ijij->ij", plain) + 1e-12)


def test_long_range_derivative_integrals_match_finite_differences() -> None:
    """∂(μν|λσ)_erf/∂A_x по центру первой функции — конечными разностями координаты."""
    molecule = _water()
    basis = build_basis("sto-3g", molecule).cartesian()
    derivative = integral_module.build_electron_repulsion_derivative(basis, molecule, 0, omega=0.3)
    # Функции кислорода (первые 5 в STO-3G) сдвигаются вместе с атомом O: производная по
    # центру μ ∈ O равна производной тензора по смещению O для тех индексов, где
    # ВСЕ четыре функции кислородные (остальные центры не меняются) — берём блок O-O-O-O
    # и делим вклад на число центров через градиент суммы: проверяем сумму по четырём
    # слотам, как это делает _orbital_gradient.
    step = 1e-4  # бор
    angstrom = step / 1.8897259886
    shifted = [
        integral_module.build_electron_repulsion(
            basis, _displaced(molecule, 0, 0, sign * angstrom), omega=0.3
        )
        for sign in (1.0, -1.0)
    ]
    numeric = (shifted[0] - shifted[1]) / (2.0 * step)
    owner = np.array([0] * 5 + [1] + [2])
    on_oxygen = owner == 0
    slots = (
        derivative,
        derivative.transpose(1, 0, 2, 3),
        derivative.transpose(2, 3, 0, 1),
        derivative.transpose(2, 3, 1, 0),
    )
    analytic = np.zeros_like(derivative)
    for slot, tensor in enumerate(slots):
        shape = [1, 1, 1, 1]
        shape[slot] = len(owner)
        analytic += on_oxygen.reshape(shape) * tensor
    assert np.max(np.abs(analytic - numeric)) < 1e-6


# --------------------------------------------------------------------------- #
# Оператор точного обмена
# --------------------------------------------------------------------------- #
def test_exact_exchange_combines_full_and_long_range_parts() -> None:
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    functional = get_functional("wb97x")
    omega, long_range = range_separation(functional)
    assert omega == pytest.approx(0.3)
    assert functional.exact_exchange_fraction + long_range == pytest.approx(1.0)
    prepared = build_integrals(basis, molecule, omega=omega)
    rng = np.random.default_rng(1)
    density = rng.normal(size=prepared.overlap.shape)
    density = density + density.T
    operator = ExactExchange.for_functional(functional, prepared)
    from quantumlab.engine.scf import exchange_matrix

    expected = functional.exact_exchange_fraction * exchange_matrix(density, prepared.eri)
    assert prepared.eri_lr is not None
    expected = expected + long_range * exchange_matrix(density, prepared.eri_lr)
    assert np.allclose(operator(density), expected, atol=1e-13)
    # Без подготовленных интегралов оператор отказывает, а не считает «обычный» гибрид.
    with pytest.raises(ValueError, match="erf"):
        ExactExchange.for_functional(functional, build_integrals(basis, molecule))


def test_global_hybrid_has_no_range_separation() -> None:
    assert range_separation(get_functional("pbe0")) == (0.0, 0.0)
    assert range_separation(get_functional("m06")) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# SCF и градиент против PySCF
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["m06", "m062x", "wb97x", "wb97x-d"])
def test_rks_energy_and_gradient_match_pyscf(name: str) -> None:
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule, GridPreset.COARSE)
    functional = get_functional(name)
    result = run_rks(basis, molecule, functional, TIGHT, grid=grid)
    assert result.converged
    reference = _pyscf_ks(molecule, 0, name, grid)
    reference.kernel()  # type: ignore[attr-defined]
    # STO-3G в PySCF и здесь отличается разрядами экспонент: ~3e-8 по энергии воды.
    assert result.total_energy == pytest.approx(reference.e_tot, abs=2e-7)  # type: ignore[attr-defined]
    gradient = rks_gradient(basis, molecule, result, grid, functional)
    method = reference.nuc_grad_method()  # type: ignore[attr-defined]
    method.grid_response = False
    assert np.max(np.abs(np.asarray(gradient.gradient) - method.kernel())) < 1e-6


@pytest.mark.parametrize("name", ["tpssh", "m062x", "wb97x"])
def test_uks_energy_and_gradient_match_pyscf_for_triplet_water(name: str) -> None:
    molecule = _water(multiplicity=3)
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule, GridPreset.COARSE)
    functional = get_functional(name)
    result = run_uks(basis, molecule, functional, TIGHT, grid=grid)
    assert result.converged
    reference = _pyscf_ks(molecule, 2, name, grid)
    reference.kernel(dm0=np.array([result.density_alpha, result.density_beta]))  # type: ignore[attr-defined]
    assert reference.converged  # type: ignore[attr-defined]
    assert result.total_energy == pytest.approx(reference.e_tot, abs=2e-7)  # type: ignore[attr-defined]
    gradient = uks_gradient(basis, molecule, result, grid, functional)
    method = reference.nuc_grad_method()  # type: ignore[attr-defined]
    method.grid_response = False
    assert np.max(np.abs(np.asarray(gradient.gradient) - method.kernel())) < 1e-6


def test_uks_m062x_on_a_radical_matches_pyscf_started_from_our_density() -> None:
    """Радикал CH: PySCF со своей догадкой уходит в другую точку, со стартом от нашей — в ту же."""
    molecule = Molecule.from_atoms(["C", "H"], [(0, 0, 0), (0, 0, 1.117)], multiplicity=2)
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule, GridPreset.COARSE)
    functional = get_functional("m062x")
    result = run_uks(basis, molecule, functional, TIGHT, grid=grid)
    assert result.converged
    reference = _pyscf_ks(molecule, 1, "m062x", grid)
    reference.kernel(dm0=np.array([result.density_alpha, result.density_beta]))  # type: ignore[attr-defined]
    assert result.total_energy == pytest.approx(reference.e_tot, abs=2e-7)  # type: ignore[attr-defined]
    gradient = uks_gradient(basis, molecule, result, grid, functional)
    method = reference.nuc_grad_method()  # type: ignore[attr-defined]
    method.grid_response = False
    assert np.max(np.abs(np.asarray(gradient.gradient) - method.kernel())) < 1e-6


def test_direct_scf_with_range_separation_equals_stored_integrals() -> None:
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule, GridPreset.COARSE)
    functional = get_functional("wb97x")
    stored = run_rks(basis, molecule, functional, TIGHT, grid=grid)
    direct = build_integrals(basis, molecule, direct=True, omega=range_separation(functional)[0])
    result = run_rks(basis, molecule, functional, TIGHT, integrals=direct, grid=grid)
    assert result.total_energy == pytest.approx(stored.total_energy, abs=1e-8)


# --------------------------------------------------------------------------- #
# Устойчивость meta-GGA и RSH
# --------------------------------------------------------------------------- #
def test_meta_gga_rks_stability_matches_pyscf() -> None:
    newton = pytest.importorskip("pyscf.soscf.newton_ah")
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule, GridPreset.COARSE)
    prepared = build_integrals(basis, molecule)
    functional = get_functional("m06")
    result = run_rks(basis, molecule, functional, TIGHT, integrals=prepared, grid=grid)
    ours = stab.rotation_stability(
        SpinFockBuilder(basis, molecule, functional=functional, integrals=prepared, grid=grid),
        result.coefficients,
        result.coefficients,
        5,
        5,
        kind="restricted",
        prefix="RKS",
    )
    lowest = {channel.name: channel.lowest_eigenvalue for channel in ours.channels}
    reference = _pyscf_ks(molecule, 0, "m06", grid)
    reference.kernel()  # type: ignore[attr-defined]
    gradient, hop, _ = newton.gen_g_hop_rhf(
        reference,
        reference.mo_coeff,  # type: ignore[attr-defined]
        reference.mo_occ,  # type: ignore[attr-defined]
        with_symmetry=False,
    )
    size = gradient.size
    hessian = np.array([hop(np.eye(size)[i]).real * 2 for i in range(size)]).T
    expected = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))[0] / 4.0
    assert lowest["rks->rks"] == pytest.approx(expected, abs=2e-6)


def test_rsh_builder_energy_matches_the_scf_energy() -> None:
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule, GridPreset.COARSE)
    functional = get_functional("wb97x")
    prepared = build_integrals(basis, molecule, omega=range_separation(functional)[0])
    result = run_rks(basis, molecule, functional, TIGHT, integrals=prepared, grid=grid)
    builder = SpinFockBuilder(basis, molecule, functional=functional, integrals=prepared, grid=grid)
    half = result.density / 2.0
    energy = builder(half, half)[2]
    assert energy == pytest.approx(result.total_energy, abs=1e-8)


# --------------------------------------------------------------------------- #
# D2-CHG
# --------------------------------------------------------------------------- #
def test_d2_chg_energy_matches_a_hand_computed_pair_sum() -> None:
    """Для HF-молекулы одна пара: энергия считается вручную по формуле статьи."""
    molecule = Molecule.from_atoms(["H", "F"], [(0, 0, 0), (0, 0, 0.917)])
    c6_unit = 1.0 / (6.02214076e23 * 4.3597447222071e-18) * (10.0 / 0.529177210903) ** 6
    c6 = math.sqrt(0.14 * 0.75) * c6_unit
    r0 = (1.001 + 1.287) / 0.529177210903
    distance = 0.917 / 0.529177210903
    expected = -c6 / distance**6 / (1.0 + 6.0 * (r0 / distance) ** 12)
    contribution = dftd2_chg_contribution(molecule)
    assert contribution.energy_hartree == pytest.approx(expected, rel=1e-12)
    assert contribution.model == "d2chg"


def test_d2_chg_gradient_matches_finite_differences_and_conserves_force() -> None:
    molecule = _water()
    contribution = dftd2_chg_contribution(molecule)
    assert np.max(np.abs(contribution.gradient.sum(axis=0))) < 1e-14
    step = 1e-4  # Å
    for atom_index, axis in ((0, 2), (1, 0), (2, 1)):
        energies = [
            dftd2_chg_contribution(
                _displaced(molecule, atom_index, axis, sign * step)
            ).energy_hartree
            for sign in (1.0, -1.0)
        ]
        numeric = (energies[0] - energies[1]) / (2.0 * step * 1.8897259886)
        assert contribution.gradient[atom_index, axis] == pytest.approx(numeric, abs=1e-9)


def test_d2_chg_refuses_elements_without_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Элемента нет в таблице — отказ, а не молчаливый нулевой вклад."""
    from quantumlab.engine import dispersion_d2

    assert 1 in chg_elements() and 54 in chg_elements()
    monkeypatch.delitem(dispersion_d2._PARAMETERS, 8)
    with pytest.raises(ValueError, match="D2"):
        dftd2_chg_contribution(_water())


# --------------------------------------------------------------------------- #
# Движок
# --------------------------------------------------------------------------- #
def _spec(
    functional: str, dispersion: DispersionCorrection = DispersionCorrection.NONE
) -> CalculationSpec:
    return CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(
            theory=TheoryFamily.DFT,
            basis="sto-3g",
            functional=functional,
            spin=SpinTreatment.RHF,
            dispersion=dispersion,
        ),
        scf=ScfSpec(),
        grid=GridSpec(preset=GridPreset.COARSE),
    )


def test_engine_adds_the_intrinsic_dispersion_of_wb97x_d() -> None:
    molecule = _water()
    engine = ReferenceEngine()
    plain = engine.run(EngineRequest(job_id="a", spec=_spec("wb97x"), molecule=molecule, threads=1))
    with_d = engine.run(
        EngineRequest(job_id="b", spec=_spec("wb97x-d"), molecule=molecule, threads=1)
    )
    assert with_d.converged and plain.converged
    assert with_d.dispersion_energy_hartree == pytest.approx(
        dftd2_chg_contribution(molecule).energy_hartree, abs=1e-12
    )
    assert plain.dispersion_energy_hartree is None
    assert all(check.verdict.value != "fail" for check in with_d.quality_checks)


def test_engine_rejects_a_second_dispersion_on_top_of_wb97x_d() -> None:
    with pytest.raises(CombinationUnavailableError):
        ReferenceEngine().assert_supported(_spec("wb97x-d", DispersionCorrection.D3_BJ))


@pytest.mark.parametrize("functional", ["m06", "m062x", "wb97x"])
def test_engine_uks_and_stability_work_for_the_new_functionals(functional: str) -> None:
    molecule = _water(multiplicity=3)
    spec = _spec(functional)
    assert spec.method is not None
    spec = spec.model_copy(
        update={
            "method": spec.method.model_copy(update={"spin": SpinTreatment.UHF}),
            "scf": ScfSpec(stability_analysis=True),
        }
    )
    result = ReferenceEngine().run(
        EngineRequest(job_id="u", spec=spec, molecule=molecule, threads=1)
    )
    assert result.converged
    names = {check.name_key for check in result.quality_checks}
    assert "wfn_stability" in names
    assert all(check.verdict.value != "fail" for check in result.quality_checks)
