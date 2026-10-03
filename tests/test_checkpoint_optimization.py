"""Контрольные точки ROHF и оптимизации геометрии.

Главное свойство контрольной точки оптимизации — **та же траектория**:
продолженный расчёт обязан прийти в ту же геометрию с тем же числом шагов, что
и непрерывный. Приблизительное совпадение здесь недостаточно: потерянный гессиан
BFGS давал бы ту же конечную точку, но другой путь (и это видно по журналу).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    OptimizationSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.optimizer import OptimizationSettings, OptimizerState, optimize_geometry
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import ScfSettings, build_integrals, run_rohf, run_uhf
from quantumlab.errors import JobCheckpointInvalidError

FIXTURES = Path(__file__).parent / "fixtures"


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def _radical() -> Molecule:
    return Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), name="ch", multiplicity=2
    )


def _run(molecule: Molecule, spec: CalculationSpec, checkpoint: str | None = None):  # type: ignore[no-untyped-def]
    payloads: list[str] = []
    result = ReferenceEngine().run(
        EngineRequest(job_id="job", molecule=molecule, spec=spec, checkpoint=checkpoint),
        checkpoint_sink=payloads.append,
    )
    return result, payloads


# --------------------------------------------------------------------------- #
# ROHF
# --------------------------------------------------------------------------- #
def _rohf_spec() -> CalculationSpec:
    return CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.ROHF),
    )


def test_rohf_restart_reproduces_the_energy_in_one_iteration() -> None:
    molecule = _radical()
    first, payloads = _run(molecule, _rohf_spec())
    assert len(payloads) == 1
    data = json.loads(payloads[0])
    assert "density_alpha" in data
    assert data["n_alpha"] == data["n_beta"] + 1
    second, _ = _run(molecule, _rohf_spec(), checkpoint=payloads[0])
    assert second.converged
    assert second.energy_hartree == pytest.approx(first.energy_hartree, abs=1e-9)
    assert second.scf_iterations < first.scf_iterations


def test_rohf_refuses_a_closed_shell_checkpoint_for_an_open_shell_molecule() -> None:
    from quantumlab.domain.spec import SpinTreatment as Spin

    closed_spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=Spin.RHF),
    )
    _, payloads = _run(_radical().model_copy(update={"multiplicity": 2}), _rohf_spec())
    water_payload = _run(_water(), closed_spec)[1][0]
    with pytest.raises(JobCheckpointInvalidError):
        _run(_radical(), _rohf_spec(), checkpoint=water_payload)
    assert payloads


def test_rohf_rejects_densities_that_are_not_nested() -> None:
    """Плотности UHF-решения не имеют общего занятого пространства — рестарт ROHF отказывает."""
    molecule = _radical()
    basis = build_basis("sto-3g", molecule)
    prepared = build_integrals(basis, molecule)
    uhf = run_uhf(basis, molecule, ScfSettings(), integrals=prepared)
    with pytest.raises(ValueError, match="ROHF"):
        run_rohf(
            basis,
            molecule,
            ScfSettings(),
            integrals=prepared,
            initial_densities=(uhf.density_alpha, uhf.density_beta),
        )


# --------------------------------------------------------------------------- #
# Оптимизатор
# --------------------------------------------------------------------------- #
def _quadratic(molecule: Molecule) -> tuple[float, np.ndarray]:
    """Гармоническая поверхность по межатомным расстояниям (инвариантна к сдвигу и повороту).

    Разные жёсткости и равновесные длины, плюс ангармонический член: BFGS
    обязан копить гессиан, а не угадывать его с первого шага.
    """
    x = np.array([atom.position for atom in molecule.atoms]) / 0.529177210903
    energy = 0.0
    gradient = np.zeros_like(x)
    constants = {(0, 1): 0.6, (0, 2): 0.45, (1, 2): 0.3}
    lengths = {(0, 1): 1.9, (0, 2): 1.7, (1, 2): 2.6}
    for (i, j), k in constants.items():
        vector = x[i] - x[j]
        distance = float(np.linalg.norm(vector))
        shift = distance - lengths[(i, j)]
        energy += k * shift**2 + 0.2 * k * shift**4
        derivative = (2.0 * k * shift + 0.8 * k * shift**3) * vector / distance
        gradient[i] += derivative
        gradient[j] -= derivative
    return float(energy), gradient


def test_optimizer_resume_follows_the_same_trajectory() -> None:
    molecule = _water()
    settings = OptimizationSettings(
        max_steps=80, max_force=1e-6, rms_force=1e-6, max_displacement=1e-5, rms_displacement=1e-5
    )
    states: list[OptimizerState] = []
    full = optimize_geometry(molecule, _quadratic, settings, on_state=states.append)
    assert full.converged
    assert len(states) >= 5
    middle = states[len(states) // 2]

    calls = 0

    def counted(candidate: Molecule) -> tuple[float, np.ndarray]:
        nonlocal calls
        calls += 1
        return _quadratic(candidate)

    resumed = optimize_geometry(molecule, counted, settings, resume=middle)
    assert resumed.converged
    assert resumed.steps == full.steps
    assert len(resumed.history) == len(full.history)
    assert resumed.energy_hartree == full.energy_hartree
    first = np.array([a.position for a in full.molecule.atoms])
    again = np.array([a.position for a in resumed.molecule.atoms])
    assert np.array_equal(first, again)
    # Продолжение не пересчитывает уже пройденное: энергий меньше, чем в полном пути.
    assert calls < len(states) * 2


def test_resume_without_the_hessian_would_change_the_path() -> None:
    """Контроль: сброс гессиана в единичный даёт другой журнал — тест выше что-то значит."""
    molecule = _water()
    settings = OptimizationSettings(
        max_steps=80, max_force=1e-6, rms_force=1e-6, max_displacement=1e-5, rms_displacement=1e-5
    )
    states: list[OptimizerState] = []
    full = optimize_geometry(molecule, _quadratic, settings, on_state=states.append)
    middle = states[2]
    reset = OptimizerState(
        step_index=middle.step_index,
        coordinates=middle.coordinates,
        energy_hartree=middle.energy_hartree,
        gradient=middle.gradient,
        hessian=np.eye(middle.hessian.shape[0]),
        previous_step=None,
        previous_gradient=None,
        displacement=middle.displacement,
        history=middle.history,
    )
    other = optimize_geometry(molecule, _quadratic, settings, resume=reset)
    assert [s.energy_hartree for s in other.history] != [s.energy_hartree for s in full.history]


# --------------------------------------------------------------------------- #
# Движок: оптимизация
# --------------------------------------------------------------------------- #
def _optimization_spec(basis: str = "sto-3g", **options: object) -> CalculationSpec:
    return CalculationSpec(
        task=Task.OPTIMIZATION,
        method=MethodSpec(theory=TheoryFamily.HF, basis=basis),
        optimization=OptimizationSpec(**options),  # type: ignore[arg-type]
    )


def test_engine_optimization_resumes_from_an_intermediate_checkpoint() -> None:
    molecule = _water().model_copy()
    # Искажённая стартовая геометрия: несколько шагов, из которых можно выбрать середину.
    atoms = list(molecule.atoms)
    atoms[1] = atoms[1].model_copy(update={"position": (0.0, 0.9, 0.55)})
    molecule = molecule.model_copy(update={"atoms": tuple(atoms)})
    spec = _optimization_spec()
    full, payloads = _run(molecule, spec)
    assert full.converged
    assert len(payloads) >= 4
    middle = payloads[len(payloads) // 2]
    resumed, resumed_payloads = _run(molecule, spec, checkpoint=middle)
    assert resumed.converged
    assert resumed.optimization_steps == full.optimization_steps
    assert resumed.energy_hartree == pytest.approx(full.energy_hartree, abs=1e-9)
    assert len(resumed_payloads) < len(payloads)
    assert resumed.final_molecule is not None
    assert full.final_molecule is not None
    old = np.array([a.position for a in full.final_molecule.atoms])
    new = np.array([a.position for a in resumed.final_molecule.atoms])
    assert np.abs(old - new).max() < 1e-9


def test_engine_optimization_checkpoint_is_refused_for_a_different_job() -> None:
    molecule = _water()
    spec = _optimization_spec()
    _, payloads = _run(molecule, spec)
    middle = payloads[0]
    # Другая исходная структура.
    other = Molecule.from_atoms(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.8)])
    with pytest.raises(JobCheckpointInvalidError):
        _run(other, spec, checkpoint=middle)
    # Другой базис — градиент и гессиан относятся к другой поверхности.
    with pytest.raises(JobCheckpointInvalidError):
        _run(molecule, _optimization_spec("3-21g"), checkpoint=middle)
    # Другие критерии — отличный отпечаток спецификации.
    with pytest.raises(JobCheckpointInvalidError):
        _run(molecule, _optimization_spec(max_force=1e-5), checkpoint=middle)
    # Контрольная точка SCF не годится для оптимизации и наоборот.
    scf_spec = CalculationSpec(
        task=Task.SINGLE_POINT, method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g")
    )
    _, scf_payloads = _run(molecule, scf_spec)
    with pytest.raises(JobCheckpointInvalidError):
        _run(molecule, spec, checkpoint=scf_payloads[0])
    with pytest.raises(JobCheckpointInvalidError):
        _run(molecule, scf_spec, checkpoint=middle)


def test_corrupted_optimization_checkpoint_is_refused() -> None:
    molecule = _water()
    spec = _optimization_spec()
    _, payloads = _run(molecule, spec)
    data = json.loads(payloads[1])
    data["hessian"][0][1] += 1.0  # нарушенная симметрия
    with pytest.raises(JobCheckpointInvalidError):
        _run(molecule, spec, checkpoint=json.dumps(data))
    data = json.loads(payloads[1])
    data["history"] = data["history"][:-1]
    with pytest.raises(JobCheckpointInvalidError):
        _run(molecule, spec, checkpoint=json.dumps(data))
    with pytest.raises(JobCheckpointInvalidError):
        _run(molecule, spec, checkpoint="не json")
