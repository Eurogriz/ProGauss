"""Избыточные внутренние координаты и ограничения в оптимизации геометрии.

Оракулы — независимые от реализации:

* матрица Вильсона — центральные конечные разности значений примитивов;
* ограниченный минимум — прямой перебор оставшихся степеней свободы по одной
  энергии (симплекс-поиск Нелдера—Мида), без градиентов и без внутренних координат;
* стационарность — проекция декартова градиента на подпространство, в котором
  ограниченная координата (её производная берётся разностями) не меняется.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    CoordinateConstraint,
    MethodSpec,
    OptimizationSpec,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.constants import ANGSTROM_TO_BOHR
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.gradients import rhf_gradient
from quantumlab.engine.internals import InternalSystem, build_primitives
from quantumlab.engine.optimizer import (
    Constraint,
    OptimizationSettings,
    OptimizerState,
    optimize_geometry,
)
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import ScfSettings, run_rhf
from quantumlab.errors import JobCheckpointInvalidError, MethodNotAvailableError

FIXTURES = Path(__file__).parent / "fixtures"
BOHR = ANGSTROM_TO_BOHR


def _flat(molecule: Molecule) -> np.ndarray:
    return np.array([v * BOHR for atom in molecule.atoms for v in atom.position])


def _positions(molecule: Molecule) -> np.ndarray:
    return np.array([atom.position for atom in molecule.atoms])


WATER = Molecule.from_atoms(
    ["O", "H", "H"], [(0.0, 0.0, 0.12), (0.0, 0.8, -0.5), (0.0, -0.7, -0.45)]
)
PEROXIDE = Molecule.from_atoms(
    ["O", "O", "H", "H"],
    [(0.0, 0.7, 0.1), (0.0, -0.7, 0.1), (0.8, 0.9, -0.4), (-0.6, -0.9, -0.5)],
)
CARBON_DIOXIDE = Molecule.from_atoms(
    ["O", "C", "O"], [(0.02, 0.0, -1.20), (0.0, 0.0, 0.0), (0.0, 0.03, 1.10)]
)
ETHANE = Molecule.from_atoms(
    ["C", "C", "H", "H", "H", "H", "H", "H"],
    [
        (0, 0, 0.76),
        (0, 0, -0.76),
        (1.0, 0, 1.15),
        (-0.5, 0.87, 1.15),
        (-0.5, -0.87, 1.15),
        (-1.0, 0.2, -1.15),
        (0.5, 0.87, -1.15),
        (0.5, -0.87, -1.15),
    ],
)
DIMER = Molecule.from_atoms(
    ["O", "H", "H", "O", "H", "H"],
    [
        (0, 0, 0),
        (0.96, 0, 0),
        (-0.24, 0.93, 0),
        (0, 0, 3.0),
        (0.96, 0, 3.0),
        (-0.24, 0.93, 3.0),
    ],
)
LINEAR_ACETYLENE = Molecule.from_atoms(
    ["H", "C", "C", "H"], [(0, 0, -1.66), (0, 0, -0.6), (0, 0, 0.6), (0, 0, 1.66)]
)


# --------------------------------------------------------------------------- #
# Матрица Вильсона и набор примитивов
# --------------------------------------------------------------------------- #
def _numerical_wilson(system: InternalSystem, x: np.ndarray) -> np.ndarray:
    step = 1e-5
    matrix = np.zeros((system.size, x.size))
    for column in range(x.size):
        plus = x.copy()
        plus[column] += step
        minus = x.copy()
        minus[column] -= step
        matrix[:, column] = system.difference(system.values(plus), system.values(minus)) / (
            2 * step
        )
    return matrix


@pytest.mark.parametrize(
    ("molecule", "dof"),
    [
        (WATER, 3),
        (PEROXIDE, 6),
        (CARBON_DIOXIDE, 4),
        (LINEAR_ACETYLENE, 7),
        (ETHANE, 18),
        (DIMER, 12),
    ],
)
def test_wilson_matrix_matches_finite_differences_and_spans_all_dof(
    molecule: Molecule, dof: int
) -> None:
    system = build_primitives(molecule)
    x = _flat(molecule)
    assert np.abs(system.wilson(x) - _numerical_wilson(system, x)).max() < 1e-7
    # Ранг B — число внутренних степеней свободы 3N−6 (3N−5 для линейных).
    assert system.rank(x) == dof


def test_primitive_set_has_the_expected_composition() -> None:
    water = Counter(p.kind for p in build_primitives(WATER).primitives)
    assert water == {"stretch": 2, "bend": 1}
    peroxide = Counter(p.kind for p in build_primitives(PEROXIDE).primitives)
    assert peroxide == {"stretch": 3, "bend": 2, "torsion": 1}
    # Линейная молекула: валентных углов нет, вместо них пары линейных изгибов.
    dioxide = Counter(p.kind for p in build_primitives(CARBON_DIOXIDE).primitives)
    assert dioxide == {"stretch": 2, "linear_bend": 2}
    # Фрагменты соединяются: у димера воды есть координата между молекулами.
    dimer = build_primitives(DIMER)
    inter = [
        p for p in dimer.primitives if p.kind == "stretch" and (p.atoms[0] < 3) != (p.atoms[1] < 3)
    ]
    assert len(inter) == 1


def test_back_transformation_reaches_the_requested_internal_displacement() -> None:
    system = build_primitives(PEROXIDE)
    x = _flat(PEROXIDE)
    q = system.values(x)
    delta = np.zeros(system.size)
    wilson = system.wilson(x)
    _, projector = system.pseudo_inverse_of_gram(wilson)
    delta[:] = projector @ np.linspace(-0.08, 0.1, system.size)
    new_x, residual = system.back_transform(x, delta)
    assert residual < 1e-9
    achieved = system.difference(system.values(new_x), q)
    assert np.abs(achieved - delta).max() < 1e-8
    # Периодичность: поворот двугранного угла через ±π даёт кратчайшую разность.
    torsion = next(i for i, p in enumerate(system.primitives) if p.kind == "torsion")
    shifted = q.copy()
    shifted[torsion] += 2 * math.pi + 0.3
    assert abs(system.difference(shifted, q)[torsion] - 0.3) < 1e-12


# --------------------------------------------------------------------------- #
# Оптимизация на реальной энергии RHF/STO-3G
# --------------------------------------------------------------------------- #
def _energy(molecule: Molecule) -> float:
    basis = build_basis("sto-3g", molecule)
    return run_rhf(basis, molecule, ScfSettings()).total_energy


def _energy_and_gradient(molecule: Molecule) -> tuple[float, np.ndarray]:
    basis = build_basis("sto-3g", molecule)
    scf = run_rhf(basis, molecule, ScfSettings())
    assert scf.converged
    return scf.total_energy, rhf_gradient(basis, molecule, scf).gradient


def _nelder_mead(function, start: np.ndarray, scale: np.ndarray) -> float:  # type: ignore[no-untyped-def]
    """Симплекс-поиск минимума функции двух переменных; возвращает значение в минимуме."""
    simplex = [start, start + np.array([scale[0], 0.0]), start + np.array([0.0, scale[1]])]
    values = [function(point) for point in simplex]
    for _ in range(200):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if values[2] - values[0] < 1e-12 and max(np.abs(simplex[2] - simplex[0])) < 1e-6:
            break
        centre = (simplex[0] + simplex[1]) / 2
        reflected = centre + (centre - simplex[2])
        reflected_value = function(reflected)
        if reflected_value < values[0]:
            expanded = centre + 2 * (centre - simplex[2])
            expanded_value = function(expanded)
            if expanded_value < reflected_value:
                simplex[2], values[2] = expanded, expanded_value
            else:
                simplex[2], values[2] = reflected, reflected_value
        elif reflected_value < values[1]:
            simplex[2], values[2] = reflected, reflected_value
        else:
            contracted = centre + 0.5 * (simplex[2] - centre)
            contracted_value = function(contracted)
            if contracted_value < values[2]:
                simplex[2], values[2] = contracted, contracted_value
            else:
                for k in (1, 2):
                    simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                    values[k] = function(simplex[k])
    return float(min(values))


def _flat_surface(_molecule: Molecule) -> tuple[float, np.ndarray]:
    """Плоская поверхность: достаточно, чтобы проверить отказы до первого шага."""
    return 0.0, np.zeros((3, 3))


def _molecule_with(base: Molecule, positions: np.ndarray) -> Molecule:
    return Molecule.from_atoms(
        [atom.symbol for atom in base.atoms], [tuple(map(float, row)) for row in positions]
    )


def test_redundant_internal_matches_cartesian_minimum_in_fewer_steps() -> None:
    cartesian = optimize_geometry(WATER, _energy_and_gradient, OptimizationSettings())
    internal = optimize_geometry(
        WATER, _energy_and_gradient, OptimizationSettings(coordinates="redundant_internal")
    )
    assert cartesian.converged
    assert internal.converged
    assert internal.energy_hartree == pytest.approx(cartesian.energy_hartree, abs=1e-7)
    assert internal.steps < cartesian.steps


def test_linear_molecule_stays_linear_and_reaches_the_cartesian_energy() -> None:
    cartesian = optimize_geometry(
        CARBON_DIOXIDE, _energy_and_gradient, OptimizationSettings(max_steps=40)
    )
    internal = optimize_geometry(
        CARBON_DIOXIDE,
        _energy_and_gradient,
        OptimizationSettings(coordinates="redundant_internal", max_steps=40),
    )
    assert internal.converged
    assert internal.energy_hartree == pytest.approx(cartesian.energy_hartree, abs=1e-7)
    p = _positions(internal.molecule)
    axis = p[2] - p[0]
    axis /= np.linalg.norm(axis)
    off_axis = np.cross(p[1] - p[0], axis)
    assert np.linalg.norm(off_axis) < 2e-3


def test_bond_constraint_is_held_and_the_energy_is_the_independent_minimum() -> None:
    target = 1.2  # Å
    constraint = Constraint((0, 1), target * BOHR)
    result = optimize_geometry(
        WATER,
        _energy_and_gradient,
        OptimizationSettings(coordinates="redundant_internal", constraints=(constraint,)),
    )
    assert result.converged
    p = _positions(result.molecule)
    assert np.linalg.norm(p[0] - p[1]) == pytest.approx(target, abs=1e-8)

    # Оракул: при фиксированной длине O–H1 минимизируем энергию по двум оставшимся
    # степеням свободы (длина второй связи и угол), не обращаясь к градиентам.
    def built(parameters: np.ndarray) -> Molecule:
        r2, angle = parameters
        return Molecule.from_atoms(
            ["O", "H", "H"],
            [
                (0.0, 0.0, 0.0),
                (target, 0.0, 0.0),
                (r2 * math.cos(angle), r2 * math.sin(angle), 0.0),
            ],
        )

    reference = _nelder_mead(
        lambda v: _energy(built(v)), np.array([1.0, math.radians(100)]), np.array([0.1, 0.1])
    )
    assert result.energy_hartree == pytest.approx(reference, abs=1e-7)


def test_dihedral_constraint_is_held_and_the_gradient_is_stationary_on_the_manifold() -> None:
    angle = math.radians(90.0)
    result = optimize_geometry(
        PEROXIDE,
        _energy_and_gradient,
        OptimizationSettings(
            coordinates="redundant_internal",
            max_steps=60,
            constraints=(Constraint((2, 0, 1, 3), angle),),
        ),
    )
    assert result.converged
    system = build_primitives(PEROXIDE, extra=())
    torsion = next(p for p in system.primitives if p.kind == "torsion")
    assert sorted(torsion.atoms) == [0, 1, 2, 3]
    x = _flat(result.molecule)
    from quantumlab.engine.internals import Primitive

    constrained = InternalSystem((Primitive("torsion", (2, 0, 1, 3)),), 4)
    assert constrained.values(x)[0] == pytest.approx(angle, abs=1e-8)

    # Независимая проверка условия минимума на многообразии: градиент энергии
    # ортогонален любому смещению, не меняющему ограниченный угол и не являющемуся
    # сдвигом или поворотом всей молекулы.
    gradient = _energy_and_gradient(result.molecule)[1].reshape(-1)
    constraint_row = _numerical_wilson(constrained, x)[0]
    translations = [np.tile(np.eye(3)[k], 4) for k in range(3)]
    centre = x.reshape(4, 3) - x.reshape(4, 3).mean(axis=0)
    rotations = [np.cross(np.eye(3)[k], centre).reshape(-1) for k in range(3)]
    rigid = np.array([*translations, *rotations])
    basis_vectors = np.vstack([rigid, constraint_row])
    q, _ = np.linalg.qr(basis_vectors.T)
    residual = gradient - q @ (q.T @ gradient)
    assert np.abs(residual).max() < 5e-4

    free = optimize_geometry(
        PEROXIDE, _energy_and_gradient, OptimizationSettings(coordinates="redundant_internal")
    )
    assert result.energy_hartree > free.energy_hartree


def test_frozen_atom_does_not_move_and_matches_the_cartesian_result() -> None:
    internal = optimize_geometry(
        PEROXIDE,
        _energy_and_gradient,
        OptimizationSettings(coordinates="redundant_internal", max_steps=60, frozen_atoms=(0,)),
    )
    cartesian = optimize_geometry(
        PEROXIDE, _energy_and_gradient, OptimizationSettings(max_steps=80, frozen_atoms=(0,))
    )
    assert internal.converged
    assert internal.molecule.atoms[0].position == pytest.approx(
        PEROXIDE.atoms[0].position, abs=1e-10
    )
    assert internal.energy_hartree == pytest.approx(cartesian.energy_hartree, abs=1e-6)


def test_holding_the_initial_value_keeps_it() -> None:
    initial = np.linalg.norm(_positions(WATER)[0] - _positions(WATER)[2])
    result = optimize_geometry(
        WATER,
        _energy_and_gradient,
        OptimizationSettings(
            coordinates="redundant_internal", constraints=(Constraint((0, 2), None),)
        ),
    )
    p = _positions(result.molecule)
    assert np.linalg.norm(p[0] - p[2]) == pytest.approx(initial, abs=1e-8)


def test_a_violated_start_is_repaired_and_a_satisfied_constraint_does_not_disturb() -> None:
    # Угол задан сильно отличным от стартового: нарушение исправляется шагами.
    result = optimize_geometry(
        WATER,
        _energy_and_gradient,
        OptimizationSettings(
            coordinates="redundant_internal",
            constraints=(Constraint((1, 0, 2), math.radians(120.0)),),
        ),
    )
    assert result.converged
    p = _positions(result.molecule)
    u, v = p[1] - p[0], p[2] - p[0]
    cosine = float(u @ v) / float(np.linalg.norm(u) * np.linalg.norm(v))
    assert math.degrees(math.acos(cosine)) == pytest.approx(120.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Ошибочные запросы
# --------------------------------------------------------------------------- #
def test_invalid_constraints_are_rejected_with_a_reason() -> None:
    settings = OptimizationSettings(coordinates="redundant_internal")

    def run(constraint: Constraint) -> None:
        optimize_geometry(
            WATER,
            _flat_surface,
            OptimizationSettings(coordinates=settings.coordinates, constraints=(constraint,)),
        )

    with pytest.raises(IndexError):
        run(Constraint((0, 7), 1.0))
    with pytest.raises(ValueError, match="повторя"):
        run(Constraint((0, 0), 1.0))
    with pytest.raises(ValueError, match="интервале"):
        run(Constraint((1, 0, 2), math.radians(180.0)))
    with pytest.raises(ValueError, match="2, 3 или 4"):
        run(Constraint((0, 1, 2, 0, 1), 1.0))
    with pytest.raises(ValueError, match="redundant_internal"):
        optimize_geometry(
            WATER,
            _flat_surface,
            OptimizationSettings(constraints=(Constraint((0, 1), 1.0),)),
        )


def test_spec_requires_a_definite_constraint() -> None:
    with pytest.raises(ValueError, match="value"):
        OptimizationSpec(constraints=(CoordinateConstraint(atoms=(0, 1)),))
    with pytest.raises(ValueError, match="одновременно"):
        OptimizationSpec(constraints=(CoordinateConstraint(atoms=(0, 1), value=1.0, frozen=True),))


def _spec(**options: object) -> CalculationSpec:
    return CalculationSpec(
        task=Task.OPTIMIZATION,
        method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
        optimization=OptimizationSpec(**options),  # type: ignore[arg-type]
    )


def _run(spec: CalculationSpec, checkpoint: str | None = None):  # type: ignore[no-untyped-def]
    payloads: list[str] = []
    result = ReferenceEngine().run(
        EngineRequest(job_id="job", molecule=WATER, spec=spec, checkpoint=checkpoint),
        checkpoint_sink=payloads.append,
    )
    return result, payloads


def test_engine_converts_units_and_refuses_constraints_in_cartesian_coordinates() -> None:
    spec = _spec(
        coordinates="redundant_internal",
        constraints=(CoordinateConstraint(atoms=(0, 1), value=1.15),),
    )
    result, _ = _run(spec)
    assert result.converged
    assert result.final_molecule is not None
    p = _positions(result.final_molecule)
    assert np.linalg.norm(p[0] - p[1]) == pytest.approx(1.15, abs=1e-7)

    cartesian = _spec(
        coordinates="cartesian", constraints=(CoordinateConstraint(atoms=(0, 1), value=1.15),)
    )
    with pytest.raises(MethodNotAvailableError):
        ReferenceEngine().assert_supported(cartesian)


def test_redundant_internal_optimization_resumes_along_the_same_trajectory() -> None:
    spec = _spec(
        coordinates="redundant_internal",
        constraints=(CoordinateConstraint(atoms=(1, 0, 2), value=110.0),),
    )
    full, payloads = _run(spec)
    assert full.converged
    assert len(payloads) >= 3
    data = json.loads(payloads[1])
    assert len(data["hessian"]) == build_primitives(WATER).size
    resumed, resumed_payloads = _run(spec, checkpoint=payloads[1])
    assert resumed.optimization_steps == full.optimization_steps
    assert resumed.energy_hartree == pytest.approx(full.energy_hartree, abs=1e-9)
    assert len(resumed_payloads) < len(payloads)
    # Чужой набор ограничений — другой отпечаток спецификации.
    other = _spec(
        coordinates="redundant_internal",
        constraints=(CoordinateConstraint(atoms=(1, 0, 2), value=100.0),),
    )
    with pytest.raises(JobCheckpointInvalidError):
        _run(other, checkpoint=payloads[1])
    # Контрольная точка декартовой оптимизации не годится для внутренних координат.
    _, cartesian_payloads = _run(_spec())
    with pytest.raises(JobCheckpointInvalidError):
        _run(_spec(coordinates="redundant_internal"), checkpoint=cartesian_payloads[1])


def test_optimizer_state_hessian_lives_in_internal_coordinates() -> None:
    states: list[OptimizerState] = []
    optimize_geometry(
        WATER,
        _energy_and_gradient,
        OptimizationSettings(coordinates="redundant_internal"),
        on_state=states.append,
    )
    size = build_primitives(WATER).size
    assert all(state.hessian.shape == (size, size) for state in states)
    assert states[-1].coordinates.shape == (9,)
