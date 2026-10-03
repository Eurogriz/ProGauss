"""Общий механизм устойчивости: DFT, ROHF и комплексные вращения.

Проверки идут от независимых оракулов:

* HF: конечная разность орбитального градиента против **аналитических**
  ``A ± B`` (включая мнимый канал ``A − B``);
* ROHF: гессиан против **вторых разностей энергии** (энергия считается тем же
  построителем, но без производных, поэтому ошибка формулы градиента не скрыта);
  замкнутая оболочка ROHF обязана дать гессиан RHF;
* DFT: против PySCF (``newton_ah`` и ``stability._gen_hop_rhf_external``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.result import QualityVerdict
from quantumlab.domain.spec import (
    CalculationSpec,
    GridPreset,
    GridSpec,
    MethodSpec,
    ScfSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine import stability as stab
from quantumlab.engine.basis import build_basis
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.dft import run_rks, run_uks
from quantumlab.engine.fock import SpinFockBuilder
from quantumlab.engine.functional import get_functional
from quantumlab.engine.quadrature import build_grid
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import (
    ScfSettings,
    build_integrals,
    run_rhf,
    run_rohf,
    run_uhf,
    spin_population,
)

FIXTURES = Path(__file__).parent / "fixtures"
TIGHT = ScfSettings(energy_tolerance=1e-12, density_tolerance=1e-10, max_iterations=300)


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def _radical() -> Molecule:
    return Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), name="ch", multiplicity=2
    )


def _h2(distance: float) -> Molecule:
    return Molecule.from_atoms(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, distance)])


def _channels(result: stab.StabilityResult) -> dict[str, float]:
    return {channel.name: channel.lowest_eigenvalue for channel in result.channels}


# --------------------------------------------------------------------------- #
# HF: конечная разность против аналитических A ± B
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "molecule_factory", [_water, lambda: _h2(2.5)], ids=["water", "h2-stretched"]
)
def test_rhf_finite_difference_matches_analytic_channels(
    molecule_factory: Callable[[], Molecule],
) -> None:
    molecule = molecule_factory()
    basis = build_basis("6-31g", molecule)
    prepared = build_integrals(basis, molecule)
    result = run_rhf(basis, molecule, TIGHT, integrals=prepared)
    n_occupied = molecule.n_electrons // 2
    analytic = _channels(
        stab.rhf_stability(
            result.coefficients, result.orbital_energies, np.asarray(prepared.eri), n_occupied
        )
    )
    numeric = _channels(
        stab.rotation_stability(
            SpinFockBuilder(basis, molecule, integrals=prepared),
            result.coefficients,
            result.coefficients,
            n_occupied,
            n_occupied,
            kind="restricted",
            prefix="RHF",
        )
    )
    assert set(analytic) == {"rhf->rhf", "rhf->uhf", "rhf->rhf:complex"}
    assert set(numeric) == set(analytic)
    for name, value in analytic.items():
        assert numeric[name] == pytest.approx(value, abs=5e-6), name


def test_uhf_finite_difference_matches_analytic_channels() -> None:
    molecule = _radical()
    basis = build_basis("6-31g", molecule)
    prepared = build_integrals(basis, molecule)
    result = run_uhf(basis, molecule, TIGHT, integrals=prepared)
    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)
    analytic = _channels(
        stab.uhf_stability(
            result.alpha_coefficients,
            result.beta_coefficients,
            result.alpha_energies,
            result.beta_energies,
            np.asarray(prepared.eri),
            n_alpha,
            n_beta,
        )
    )
    numeric = _channels(
        stab.rotation_stability(
            SpinFockBuilder(basis, molecule, integrals=prepared),
            result.alpha_coefficients,
            result.beta_coefficients,
            n_alpha,
            n_beta,
            kind="unrestricted",
            prefix="UHF",
        )
    )
    assert set(numeric) == set(analytic) == {"uhf->uhf", "uhf->uhf:complex"}
    for name, value in analytic.items():
        assert numeric[name] == pytest.approx(value, abs=5e-6), name


def test_stretched_h2_has_a_negative_complex_free_triplet_channel() -> None:
    """Известное значение: растянутый H2 RHF→UHF (6-31G, 2.2 Å) отрицателен, мнимый — нет."""
    molecule = _h2(2.2)
    basis = build_basis("6-31g", molecule)
    prepared = build_integrals(basis, molecule)
    result = run_rhf(basis, molecule, TIGHT, integrals=prepared)
    channels = _channels(
        stab.rotation_stability(
            SpinFockBuilder(basis, molecule, integrals=prepared),
            result.coefficients,
            result.coefficients,
            1,
            1,
            kind="restricted",
            prefix="RHF",
        )
    )
    assert channels["rhf->uhf"] < -0.2
    assert channels["rhf->rhf"] > 0.0


# --------------------------------------------------------------------------- #
# ROHF
# --------------------------------------------------------------------------- #
def _rohf_setup() -> tuple[SpinFockBuilder, np.ndarray, tuple[np.ndarray, np.ndarray], int, int]:
    molecule = _radical()
    basis = build_basis("6-31g", molecule)
    prepared = build_integrals(basis, molecule)
    result = run_rohf(basis, molecule, TIGHT, integrals=prepared)
    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)
    n_orbitals = result.coefficients.shape[1]
    occupations = (
        (np.arange(n_orbitals) < n_alpha).astype(float),
        (np.arange(n_orbitals) < n_beta).astype(float),
    )
    builder = SpinFockBuilder(basis, molecule, integrals=prepared)
    return builder, result.coefficients, occupations, n_alpha, n_beta


def test_rohf_hessian_matches_second_differences_of_the_energy() -> None:
    builder, coefficients, occupations, n_alpha, n_beta = _rohf_setup()
    n_orbitals = coefficients.shape[1]
    space = stab.rohf_space(n_alpha, n_beta, n_orbitals)
    hessian = stab.rotation_hessian(builder, (coefficients, coefficients), occupations, space)

    def energy(parameters: np.ndarray) -> float:
        generator = np.zeros((n_orbitals, n_orbitals))
        np.add.at(generator, (space.p, space.q), parameters)
        np.add.at(generator, (space.q, space.p), -parameters)
        rotated = coefficients @ stab._unitary(generator, imaginary=False)
        densities = [(rotated * n) @ rotated.T for n in occupations]
        return builder(densities[0], densities[1])[2]

    rng = np.random.default_rng(3)
    step = 1e-3
    for _ in range(4):
        direction = rng.normal(size=space.size)
        direction /= np.linalg.norm(direction)
        second = (
            energy(step * direction) + energy(-step * direction) - 2.0 * energy(0.0 * direction)
        ) / step**2
        assert direction @ hessian @ direction == pytest.approx(second, rel=1e-4)


def test_rohf_stationarity_gradient_vanishes() -> None:
    builder, coefficients, occupations, n_alpha, n_beta = _rohf_setup()
    space = stab.rohf_space(n_alpha, n_beta, coefficients.shape[1])
    gradient = stab._rotation_gradient(
        builder,
        (coefficients, coefficients),
        occupations,
        space,
        np.zeros(space.size),
        imaginary=False,
    )
    assert np.abs(gradient).max() < 1e-8


def test_closed_shell_rohf_reproduces_the_rhf_hessian() -> None:
    molecule = _water()
    basis = build_basis("6-31g", molecule)
    prepared = build_integrals(basis, molecule)
    rhf = run_rhf(basis, molecule, TIGHT, integrals=prepared)
    rohf = run_rohf(basis, molecule, TIGHT, integrals=prepared)
    builder = SpinFockBuilder(basis, molecule, integrals=prepared)
    singlet = stab.rotation_stability(
        builder, rhf.coefficients, rhf.coefficients, 5, 5, kind="restricted", prefix="RHF"
    )
    restricted_open = stab.rotation_stability(
        builder, rohf.coefficients, rohf.coefficients, 5, 5, kind="rohf", prefix="ROHF"
    )
    assert _channels(restricted_open)["rohf->rohf"] == pytest.approx(
        _channels(singlet)["rhf->rhf"], abs=1e-6
    )
    assert _channels(restricted_open)["rohf->rohf:complex"] == pytest.approx(
        _channels(singlet)["rhf->rhf:complex"], abs=1e-6
    )


# --------------------------------------------------------------------------- #
# DFT
# --------------------------------------------------------------------------- #
def test_dft_builder_energy_matches_the_scf_energy() -> None:
    """Построитель фокиана — не копия логики на слух: энергия совпадает с SCF."""
    molecule = _radical()
    basis = build_basis("sto-3g", molecule)
    prepared = build_integrals(basis, molecule)
    functional = get_functional("pbe0")
    grid = build_grid(molecule, GridPreset.COARSE)
    result = run_uks(basis, molecule, functional, TIGHT, integrals=prepared, grid=grid)
    builder = SpinFockBuilder(basis, molecule, functional=functional, integrals=prepared, grid=grid)
    energy = builder(result.density_alpha, result.density_beta)[2]
    assert energy == pytest.approx(result.total_energy, abs=1e-9)


def test_dft_complex_channel_equals_delta_epsilon_minus_scaled_exchange() -> None:
    """Для гибрида ``A−B = Δε − α[(ij|ab) − (ib|ja)]``: XC-ядро в нём не участвует."""
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    prepared = build_integrals(basis, molecule)
    functional = get_functional("pbe0")
    grid = build_grid(molecule, GridPreset.COARSE)
    result = run_rks(basis, molecule, functional, TIGHT, integrals=prepared, grid=grid)
    builder = SpinFockBuilder(basis, molecule, functional=functional, integrals=prepared, grid=grid)
    numeric = _channels(
        stab.rotation_stability(
            builder, result.coefficients, result.coefficients, 5, 5, kind="restricted", prefix="RKS"
        )
    )
    eri = np.asarray(prepared.eri)
    occ, vir = result.coefficients[:, :5], result.coefficients[:, 5:]
    ibja, ijab = stab._exchange_blocks(eri, occ, vir)
    delta = np.diag(stab._delta_epsilon(np.asarray(result.orbital_energies), 5))
    alpha = functional.exact_exchange_fraction
    expected = float(np.linalg.eigvalsh(delta + alpha * (ibja - ijab))[0])
    assert numeric["rks->rks:complex"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.scientific
def test_rks_stability_matches_pyscf() -> None:
    pyscf = pytest.importorskip("pyscf")
    pyscf_stability = pytest.importorskip("pyscf.scf.stability")
    newton = pytest.importorskip("pyscf.soscf.newton_ah")
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    prepared = build_integrals(basis, molecule)
    functional = get_functional("pbe0")
    grid = build_grid(molecule)
    result = run_rks(basis, molecule, functional, TIGHT, integrals=prepared, grid=grid)
    ours = _channels(
        stab.rotation_stability(
            SpinFockBuilder(basis, molecule, functional=functional, integrals=prepared, grid=grid),
            result.coefficients,
            result.coefficients,
            5,
            5,
            kind="restricted",
            prefix="RKS",
        )
    )

    xyz = (FIXTURES / "water.xyz").read_text(encoding="utf-8").split("\n", 2)[2]
    mol = pyscf.gto.M(atom=xyz, basis="sto-3g", cart=True, unit="Angstrom", verbose=0)
    mf = mol.RKS()
    mf.xc = "pbe0"
    mf.grids.level = 5
    mf.run(conv_tol=1e-12)
    gradient, hop, _ = newton.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    size = gradient.size
    internal = np.array([hop(np.eye(size)[i]).real * 2 for i in range(size)]).T
    _, _, external_hop, _ = pyscf_stability._gen_hop_rhf_external(mf)
    external = np.array([external_hop(np.eye(size)[i]) for i in range(size)]).T
    assert ours["rks->rks"] == pytest.approx(
        np.linalg.eigvalsh(0.5 * (internal + internal.T))[0] / 4.0, abs=2e-6
    )
    assert ours["rks->uks"] == pytest.approx(
        np.linalg.eigvalsh(0.5 * (external + external.T))[0], abs=2e-6
    )


@pytest.mark.scientific
def test_uks_internal_stability_matches_pyscf() -> None:
    pyscf = pytest.importorskip("pyscf")
    newton = pytest.importorskip("pyscf.soscf.newton_ah")
    molecule = _radical()
    basis = build_basis("sto-3g", molecule)
    prepared = build_integrals(basis, molecule)
    functional = get_functional("pbe0")
    grid = build_grid(molecule)
    result = run_uks(basis, molecule, functional, TIGHT, integrals=prepared, grid=grid)
    ours = _channels(
        stab.rotation_stability(
            SpinFockBuilder(basis, molecule, functional=functional, integrals=prepared, grid=grid),
            result.alpha_coefficients,
            result.beta_coefficients,
            4,
            3,
            kind="unrestricted",
            prefix="UKS",
        )
    )
    mol = pyscf.gto.M(
        atom="C 0 0 0; H 0 0 1.117", basis="sto-3g", cart=True, unit="Angstrom", spin=1, verbose=0
    )
    mf = mol.UKS()
    mf.xc = "pbe0"
    mf.grids.level = 5
    mf.run(conv_tol=1e-12)
    gradient, hop, _ = newton.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    size = gradient.size
    hessian = np.array([hop(np.eye(size)[i]).real for i in range(size)]).T
    reference = float(np.linalg.eigvalsh(0.5 * (hessian + hessian.T))[0])
    assert ours["uks->uks"] == pytest.approx(reference, abs=5e-6)


# --------------------------------------------------------------------------- #
# Движок
# --------------------------------------------------------------------------- #
def _spec(
    theory: TheoryFamily,
    spin: SpinTreatment,
    functional: str | None = None,
    preset: GridPreset = GridPreset.COARSE,
) -> CalculationSpec:
    return CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=theory, functional=functional, basis="sto-3g", spin=spin),
        scf=ScfSpec(stability_analysis=True),
        grid=GridSpec(preset=preset),
    )


@pytest.mark.parametrize(
    ("molecule_factory", "theory", "spin", "functional", "expected"),
    [
        (_water, TheoryFamily.DFT, SpinTreatment.RHF, "pbe", "rks->uks"),
        (_radical, TheoryFamily.DFT, SpinTreatment.UHF, "pbe", "uks->uks:complex"),
        (_radical, TheoryFamily.HF, SpinTreatment.ROHF, None, "rohf->rohf:complex"),
    ],
    ids=["rks", "uks", "rohf"],
)
def test_engine_reports_stability_for_dft_and_rohf(
    molecule_factory: Callable[[], Molecule],
    theory: TheoryFamily,
    spin: SpinTreatment,
    functional: str | None,
    expected: str,
) -> None:
    result = ReferenceEngine().run(
        EngineRequest(
            job_id="stab",
            molecule=molecule_factory(),
            # Открытая оболочка DFT на грубой сетке не сходится из-за шума сетки
            # (так было и до этой работы), поэтому для UKS — штатная мелкая.
            spec=_spec(
                theory,
                spin,
                functional,
                GridPreset.FINE if spin is SpinTreatment.UHF else GridPreset.COARSE,
            ),
        )
    )
    check = next(c for c in result.quality_checks if c.name_key == "wfn_stability")
    assert check.verdict in (QualityVerdict.PASS, QualityVerdict.WARNING)
    assert expected in (check.detail or "")
