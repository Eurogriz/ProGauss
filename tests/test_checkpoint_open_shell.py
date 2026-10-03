"""Контрольные точки UHF, RKS и UKS: запись, рестарт и отказ при несовпадении."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.errors import JobCheckpointInvalidError

FIXTURES = Path(__file__).parent / "fixtures"


def _radical() -> Molecule:
    return Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), name="ch", multiplicity=2
    )


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def _spec(theory: TheoryFamily, spin: SpinTreatment) -> CalculationSpec:
    return CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(
            theory=theory,
            basis="sto-3g",
            spin=spin,
            functional="pbe0" if theory is TheoryFamily.DFT else None,
        ),
    )


def _run(molecule: Molecule, spec: CalculationSpec, checkpoint: str | None = None):  # type: ignore[no-untyped-def]
    payloads: list[str] = []
    result = ReferenceEngine().run(
        EngineRequest(job_id="job", molecule=molecule, spec=spec, checkpoint=checkpoint),
        checkpoint_sink=payloads.append,
    )
    return result, payloads


CASES = [
    pytest.param(_radical, TheoryFamily.HF, SpinTreatment.UHF, id="uhf"),
    pytest.param(_radical, TheoryFamily.DFT, SpinTreatment.UHF, id="uks"),
    pytest.param(_water, TheoryFamily.DFT, SpinTreatment.RHF, id="rks"),
]


@pytest.mark.parametrize(("make_molecule", "theory", "spin"), CASES)
def test_restart_from_a_converged_checkpoint_reproduces_the_energy(
    make_molecule: Callable[[], Molecule],
    theory: TheoryFamily,
    spin: SpinTreatment,
) -> None:
    molecule = make_molecule()
    spec = _spec(theory, spin)
    first, payloads = _run(molecule, spec)
    assert len(payloads) == 1
    second, _ = _run(molecule, spec, checkpoint=payloads[0])
    assert second.converged
    assert second.energy_hartree == pytest.approx(first.energy_hartree, abs=1e-8)
    # Рестарт из сошедшейся плотности обязан сходиться заметно быстрее старта с нуля.
    assert second.scf_iterations < first.scf_iterations


def test_open_shell_checkpoint_carries_spin_densities() -> None:
    _, payloads = _run(_radical(), _spec(TheoryFamily.HF, SpinTreatment.UHF))
    data = json.loads(payloads[0])
    assert data["schema_version"] == "2"
    assert data["n_alpha"] == data["n_beta"] + 1
    assert "density_alpha" in data
    assert "density_beta" in data


def test_restricted_checkpoint_without_spin_densities_is_refused_by_uhf() -> None:
    water = _water()
    _, payloads = _run(water, _spec(TheoryFamily.HF, SpinTreatment.RHF))
    with pytest.raises(JobCheckpointInvalidError):
        _run(water, _spec(TheoryFamily.HF, SpinTreatment.UHF), checkpoint=payloads[0])


def test_corrupted_spin_density_is_rejected() -> None:
    radical = _radical()
    spec = _spec(TheoryFamily.HF, SpinTreatment.UHF)
    _, payloads = _run(radical, spec)
    data = json.loads(payloads[0])
    del data["density_beta"]
    with pytest.raises(JobCheckpointInvalidError):
        _run(radical, spec, checkpoint=json.dumps(data))
    data = json.loads(payloads[0])
    data["density_alpha"][0][0] += 0.5
    with pytest.raises(JobCheckpointInvalidError):
        _run(radical, spec, checkpoint=json.dumps(data))
