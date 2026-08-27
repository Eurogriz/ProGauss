"""Отпечаток расчёта — основа воспроизводимости (§40 ТЗ)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantumlab.domain.fingerprint import build_fingerprint
from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.domain.spec import CalculationSpec, MethodSpec, PrecisionProfile, Task, TheoryFamily

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def spec(basis: str = "def2-svp") -> CalculationSpec:
    return CalculationSpec(
        task=Task.OPTIMIZATION,
        profile=PrecisionProfile.STANDARD,
        method=MethodSpec(theory=TheoryFamily.DFT, functional="pbe0", basis=basis),
    )


def test_fingerprint_is_deterministic(water: Molecule) -> None:
    first = build_fingerprint(
        spec=spec(), molecule=water, software_version="0.1.0", engine_version="0.1.0"
    )
    second = build_fingerprint(
        spec=spec(), molecule=water, software_version="0.1.0", engine_version="0.1.0"
    )
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert len(first.short) == 12


def test_fingerprint_reacts_to_every_component(water: Molecule) -> None:
    baseline = build_fingerprint(
        spec=spec(), molecule=water, software_version="0.1.0", engine_version="0.1.0"
    )
    changed_basis = build_fingerprint(
        spec=spec("def2-tzvp"), molecule=water, software_version="0.1.0", engine_version="0.1.0"
    )
    moved = Molecule(
        name="water", atoms=(Atom(symbol="O", position=(0.0, 0.0, 0.01)), *water.atoms[1:])
    )
    changed_geometry = build_fingerprint(
        spec=spec(), molecule=moved, software_version="0.1.0", engine_version="0.1.0"
    )
    changed_version = build_fingerprint(
        spec=spec(), molecule=water, software_version="0.2.0", engine_version="0.1.0"
    )
    changed_engine = build_fingerprint(
        spec=spec(), molecule=water, software_version="0.1.0", engine_version="0.2.0"
    )
    digests = {
        baseline.digest,
        changed_basis.digest,
        changed_geometry.digest,
        changed_version.digest,
        changed_engine.digest,
    }
    assert len(digests) == 5


def test_hardware_and_environment_change_the_fingerprint(water: Molecule) -> None:
    baseline = build_fingerprint(
        spec=spec(), molecule=water, software_version="0.1.0", engine_version="0.1.0"
    )
    with_hardware = build_fingerprint(
        spec=spec(),
        molecule=water,
        software_version="0.1.0",
        engine_version="0.1.0",
        hardware={"cpu": "EPYC 9654", "cores": "96"},
        environment={"blas": "openblas"},
    )
    assert baseline.digest != with_hardware.digest
    assert baseline.components["hardware"] != with_hardware.components["hardware"]


def test_final_geometry_is_part_of_completed_fingerprint(water: Molecule) -> None:
    optimized = Molecule(
        name="water", atoms=(*water.atoms[:2], Atom(symbol="H", position=(0.0, 0.0, 1.0)))
    )
    without_final = build_fingerprint(
        spec=spec(), molecule=water, software_version="0.1.0", engine_version="0.1.0"
    )
    with_final = build_fingerprint(
        spec=spec(),
        molecule=water,
        software_version="0.1.0",
        engine_version="0.1.0",
        final_molecule=optimized,
    )
    assert without_final.digest != with_final.digest
    assert with_final.components["final_structure"] == optimized.structure_hash()
