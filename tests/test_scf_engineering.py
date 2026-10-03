"""EDIIS, скрининг Шварца и прямой SCF: энергии обязаны совпасть с обычным путём.

Оракулом служит тот же движок без новой возможности: EDIIS и direct SCF не
меняют решаемое уравнение, поэтому любое расхождение энергии — ошибка, а не
«допустимая разница методов».
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    ScfSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.dft import run_rks, run_uks
from quantumlab.engine.functional import get_functional
from quantumlab.engine.integrals import (
    DirectEri,
    build_electron_repulsion,
    schwarz_bounds,
)
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import (
    ScfSettings,
    build_integrals,
    run_rhf,
    run_uhf,
)
from quantumlab.errors import CombinationUnavailableError

FIXTURES = Path(__file__).parent / "fixtures"


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def _radical() -> Molecule:
    return Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), name="ch", multiplicity=2
    )


def _stretched_water() -> Molecule:
    return Molecule.from_atoms(["O", "H", "H"], [(0, 0, 0), (0, 1.9, 0.0), (1.9, 0, 0)])


def _water_pair(separation: float) -> Molecule:
    species: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for k in range(2):
        for symbol, (x, y, z) in (
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.96, 0.0, 0.0)),
            ("H", (-0.24, 0.93, 0.0)),
        ):
            species.append(symbol)
            coordinates.append((x + k * separation, y, z))
    return Molecule.from_atoms(species, coordinates)


# --------------------------------------------------------------------------- #
# EDIIS
# --------------------------------------------------------------------------- #
def test_ediis_rhf_reaches_the_diis_energy_on_a_stretched_molecule() -> None:
    molecule = _stretched_water()
    basis = build_basis("6-31g", molecule)
    plain = run_rhf(basis, molecule, ScfSettings(max_iterations=200))
    ediis = run_rhf(basis, molecule, ScfSettings(ediis=True, max_iterations=200))
    assert plain.converged
    assert ediis.converged
    assert ediis.total_energy == pytest.approx(plain.total_energy, abs=1e-8)
    assert "ediis" in ediis.strategies_used
    # Растянутая молекула — тот случай, ради которого EDIIS существует.
    assert ediis.iterations < plain.iterations


def test_ediis_uhf_matches_the_diis_energy() -> None:
    molecule = _radical()
    basis = build_basis("6-31g", molecule)
    plain = run_uhf(basis, molecule, ScfSettings(max_iterations=200))
    ediis = run_uhf(basis, molecule, ScfSettings(ediis=True, max_iterations=200))
    assert ediis.converged
    assert ediis.total_energy == pytest.approx(plain.total_energy, abs=1e-8)


def test_ediis_dft_matches_the_diis_energy() -> None:
    water = _water()
    basis = build_basis("sto-3g", water)
    functional = get_functional("pbe")
    plain = run_rks(basis, water, functional, ScfSettings())
    ediis = run_rks(basis, water, functional, ScfSettings(ediis=True))
    assert ediis.total_energy == pytest.approx(plain.total_energy, abs=1e-8)

    radical = _radical()
    basis = build_basis("sto-3g", radical)
    plain_u = run_uks(basis, radical, functional, ScfSettings())
    ediis_u = run_uks(basis, radical, functional, ScfSettings(ediis=True))
    assert ediis_u.total_energy == pytest.approx(plain_u.total_energy, abs=1e-8)


def test_ediis_is_refused_for_rohf() -> None:
    spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.ROHF),
        scf=ScfSpec(fallback_strategies=("diis", "ediis")),
    )
    with pytest.raises(CombinationUnavailableError):
        ReferenceEngine().assert_supported(spec)


# --------------------------------------------------------------------------- #
# Скрининг Шварца и потоки
# --------------------------------------------------------------------------- #
def test_screened_eri_differs_from_the_full_tensor_by_less_than_the_threshold() -> None:
    molecule = _water_pair(7.0)
    basis = build_basis("sto-3g", molecule)
    full = build_electron_repulsion(basis, molecule)
    screened = build_electron_repulsion(basis, molecule, screening=1e-10)
    assert np.abs(full - screened).max() < 1e-10
    # Пара молекул в 7 Å — скрининг действительно что-то отбрасывает.
    assert np.count_nonzero(screened) < np.count_nonzero(full)


def test_schwarz_bounds_dominate_every_integral() -> None:
    molecule = _water_pair(3.0)
    basis = build_basis("sto-3g", molecule)
    eri = build_electron_repulsion(basis, molecule)
    bounds = np.asarray(schwarz_bounds(basis, molecule, threads=1))
    # Матрица оценок на уровне функций: |(μν|λσ)| ≤ Q_μν Q_λσ.
    q = np.sqrt(np.abs(np.einsum("uvuv->uv", eri)))
    assert bounds.shape[0] > 0
    estimate = q[:, :, None, None] * q[None, None, :, :]
    assert np.all(np.abs(eri) <= estimate + 1e-12)


def test_threaded_assembly_is_identical_to_serial() -> None:
    molecule = _water()
    basis = build_basis("6-31g", molecule)
    serial = build_electron_repulsion(basis, molecule)
    threaded = build_electron_repulsion(basis, molecule, threads=2)
    assert np.array_equal(serial, threaded)


# --------------------------------------------------------------------------- #
# Прямой SCF
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("basis_name", ["6-31g", "cc-pvdz"])
def test_direct_coulomb_and_exchange_match_the_stored_tensor(basis_name: str) -> None:
    """6-31G — декартов базис, cc-pVDZ — сферический (перевод через ``T``)."""
    molecule = _water()
    basis = build_basis(basis_name, molecule)
    eri = build_electron_repulsion(basis, molecule)
    rng = np.random.default_rng(7)
    density = rng.normal(size=(basis.n_functions, basis.n_functions))
    density = density + density.T
    direct = DirectEri(basis, molecule, threshold=1e-12)
    assert np.abs(direct.coulomb(density) - np.einsum("ls,uvls->uv", density, eri)).max() < 1e-11
    assert np.abs(direct.exchange(density) - np.einsum("ls,ulvs->uv", density, eri)).max() < 1e-11


def test_direct_rhf_uhf_rks_reproduce_the_stored_energies() -> None:
    water = _water()
    basis = build_basis("6-31g", water)
    direct = build_integrals(basis, water, direct=True, screening=1e-12)
    assert isinstance(direct.eri, DirectEri)
    stored_rhf = run_rhf(basis, water)
    direct_rhf = run_rhf(basis, water, integrals=direct)
    assert direct_rhf.total_energy == pytest.approx(stored_rhf.total_energy, abs=1e-9)

    radical = _radical()
    basis = build_basis("6-31g", radical)
    direct = build_integrals(basis, radical, direct=True, screening=1e-12)
    stored_uhf = run_uhf(basis, radical)
    direct_uhf = run_uhf(basis, radical, integrals=direct)
    assert direct_uhf.total_energy == pytest.approx(stored_uhf.total_energy, abs=1e-9)

    basis = build_basis("sto-3g", water)
    direct = build_integrals(basis, water, direct=True, screening=1e-12)
    functional = get_functional("pbe0")
    stored_rks = run_rks(basis, water, functional, ScfSettings())
    direct_rks = run_rks(basis, water, functional, ScfSettings(), integrals=direct)
    assert direct_rks.total_energy == pytest.approx(stored_rks.total_energy, abs=1e-9)


def test_direct_engine_run_matches_the_stored_run() -> None:
    water = _water()
    method = MethodSpec(theory=TheoryFamily.HF, basis="sto-3g")
    plain = ReferenceEngine().run(
        EngineRequest(
            job_id="plain",
            molecule=water,
            spec=CalculationSpec(task=Task.SINGLE_POINT, method=method),
            threads=2,
        )
    )
    direct = ReferenceEngine().run(
        EngineRequest(
            job_id="direct",
            molecule=water,
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=method,
                scf=ScfSpec(direct=True, stability_analysis=True),
            ),
            threads=2,
        )
    )
    assert direct.energy_hartree == pytest.approx(plain.energy_hartree, abs=1e-9)
    # В прямом режиме тензора нет — устойчивость считается конечной разностью.
    check = next(c for c in direct.quality_checks if c.name_key == "wfn_stability")
    assert "rhf->uhf" in (check.detail or "")
    # Мнимые вращения в прямом режиме не считаются (эрмитова плотность).
    assert "complex" not in (check.detail or "")


def test_direct_eri_refuses_a_complex_density() -> None:
    water = _water()
    basis = build_basis("sto-3g", water)
    direct = DirectEri(basis, water)
    with pytest.raises(TypeError):
        direct.exchange(np.eye(basis.n_functions, dtype=complex))
