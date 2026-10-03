"""Анализ устойчивости HF: собственные значения против PySCF и поведение движка."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.result import QualityVerdict
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
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import (
    ScfSettings,
    build_integrals,
    run_rhf,
    run_uhf,
    spin_population,
)
from quantumlab.engine.stability import rhf_stability, uhf_stability
from quantumlab.errors import CombinationUnavailableError

TIGHT = ScfSettings(energy_tolerance=1e-12, density_tolerance=1e-10, max_iterations=300)
WATER = Path(__file__).parent / "fixtures" / "water.xyz"


def _h2(distance: float) -> Molecule:
    return Molecule.from_atoms(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, distance)])


def _water() -> Molecule:
    return Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")


def _rhf_channels(molecule: Molecule, basis_name: str = "6-31g") -> dict[str, float]:
    basis = build_basis(basis_name, molecule)
    prepared = build_integrals(basis, molecule)
    result = run_rhf(basis, molecule, TIGHT, integrals=prepared)
    stability = rhf_stability(
        result.coefficients,
        result.orbital_energies,
        np.asarray(prepared.eri),
        molecule.n_electrons // 2,
    )
    return {channel.name: channel.lowest_eigenvalue for channel in stability.channels}


def test_water_is_stable() -> None:
    channels = _rhf_channels(_water())
    assert channels["rhf->rhf"] > 0.0
    assert channels["rhf->uhf"] > 0.0


def test_stretched_h2_has_a_triplet_instability() -> None:
    """Растянутая связь: RHF-решение — седловая точка по спин-нарушающему вращению."""
    channels = _rhf_channels(_h2(2.5))
    assert channels["rhf->uhf"] < -0.1
    assert channels["rhf->rhf"] > 0.0


def test_equilibrium_h2_is_stable() -> None:
    assert _rhf_channels(_h2(0.74))["rhf->uhf"] > 0.0


def test_uhf_stability_reports_unstable_radical_saddle() -> None:
    """UHF из core-догадки на CH — не минимум: внутренняя неустойчивость отрицательна."""
    molecule = Molecule.from_atoms(["C", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.117)], multiplicity=2)
    basis = build_basis("sto-3g", molecule)
    prepared = build_integrals(basis, molecule)
    result = run_uhf(basis, molecule, TIGHT, integrals=prepared)
    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)
    stability = uhf_stability(
        result.alpha_coefficients,
        result.beta_coefficients,
        result.alpha_energies,
        result.beta_energies,
        np.asarray(prepared.eri),
        n_alpha,
        n_beta,
    )
    assert stability.channels[0].name == "uhf->uhf"
    assert stability.channels[0].lowest_eigenvalue == pytest.approx(-0.043166, abs=1e-5)
    assert not stability.stable


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("distance", "basis_name"), [(0.74, "6-31g"), (2.5, "6-31g"), (2.5, "sto-3g")]
)
def test_rhf_eigenvalues_match_pyscf(distance: float, basis_name: str) -> None:
    """Собственные значения совпадают с гессианом PySCF.

    Внешняя (RHF→UHF) матрица у PySCF — ровно ``A+B``; внутренняя — полный
    энергетический гессиан по параметрам вращения, то есть ``4(A+B)`` для RHF.
    """
    pyscf = pytest.importorskip("pyscf")
    stability = pytest.importorskip("pyscf.scf.stability")
    newton = pytest.importorskip("pyscf.soscf.newton_ah")
    molecule = _h2(distance)
    mol = pyscf.gto.M(
        atom=f"H 0 0 0; H 0 0 {distance}", basis=basis_name, cart=True, unit="Angstrom", verbose=0
    )
    mf = mol.RHF().run(conv_tol=1e-12)
    gradient, hop, _ = newton.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    size = gradient.size
    internal = np.array([hop(np.eye(size)[i]).real * 2 for i in range(size)]).T
    _, _, external_hop, _ = stability._gen_hop_rhf_external(mf)
    external = np.array([external_hop(np.eye(size)[i]) for i in range(size)]).T
    reference_internal = np.linalg.eigvalsh(0.5 * (internal + internal.T))[0] / 4.0
    reference_external = np.linalg.eigvalsh(0.5 * (external + external.T))[0]

    ours = _rhf_channels(molecule, basis_name)
    assert ours["rhf->rhf"] == pytest.approx(reference_internal, abs=1e-5)
    assert ours["rhf->uhf"] == pytest.approx(reference_external, abs=1e-5)


# --------------------------------------------------------------------------- #
# Движок
# --------------------------------------------------------------------------- #
def _engine_spec(
    *,
    theory: TheoryFamily = TheoryFamily.HF,
    spin: SpinTreatment = SpinTreatment.RHF,
    task: Task = Task.SINGLE_POINT,
    functional: str | None = None,
) -> CalculationSpec:
    return CalculationSpec(
        task=task,
        method=MethodSpec(theory=theory, functional=functional, basis="6-31g", spin=spin),
        scf=ScfSpec(stability_analysis=True),
    )


def _run(molecule: Molecule, spec: CalculationSpec):  # type: ignore[no-untyped-def]
    return ReferenceEngine().run(EngineRequest(job_id="job-test", molecule=molecule, spec=spec))


def test_engine_flags_the_unstable_rhf_solution() -> None:
    result = _run(_h2(2.5), _engine_spec())
    check = next(c for c in result.quality_checks if c.name_key == "wfn_stability")
    assert check.verdict is QualityVerdict.WARNING
    assert any(w.key == "warning.scf_unstable" for w in result.warnings)
    assert (
        "rhf->uhf"
        in next(w for w in result.warnings if w.key == "warning.scf_unstable").params["channels"]
    )


def test_engine_passes_a_stable_rhf_solution() -> None:
    result = _run(_water(), _engine_spec())
    check = next(c for c in result.quality_checks if c.name_key == "wfn_stability")
    assert check.verdict is QualityVerdict.PASS
    assert not any(w.key == "warning.scf_unstable" for w in result.warnings)


def test_engine_stability_for_open_shell_uses_uhf() -> None:
    radical = Molecule.from_atoms(["C", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.117)], multiplicity=2)
    spec = _engine_spec(spin=SpinTreatment.UHF)
    result = _run(radical, spec)
    check = next(c for c in result.quality_checks if c.name_key == "wfn_stability")
    assert "uhf->uhf" in (check.detail or "")


def test_stability_is_not_requested_silently_for_unsupported_combinations() -> None:
    """Отказ остаётся там, где анализа нет: meta-GGA и не-одноточечные задачи."""
    engine = ReferenceEngine()
    for spec in (
        _engine_spec(theory=TheoryFamily.DFT, functional="tpssh"),
        _engine_spec(task=Task.OPTIMIZATION),
    ):
        with pytest.raises(CombinationUnavailableError):
            engine.assert_supported(spec)
    # DFT и ROHF теперь проходят проверку допуска.
    assert engine.assert_supported(_engine_spec(theory=TheoryFamily.DFT, functional="pbe"))
    assert engine.assert_supported(_engine_spec(spin=SpinTreatment.ROHF))
