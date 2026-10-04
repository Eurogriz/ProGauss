"""Оптимизация геометрии в избыточных внутренних координатах с ограничениями.

Метод Пэна—Айалы—Шлегеля—Фриша (Peng, Ayala, Schlegel, Frisch, J. Comput.
Chem. 17, 49, 1996) с проекцией ограничений:

1. Градиент переводится во внутренние координаты ``g_q = G⁻ B g_x``
   (``G = B Bᵀ``, ``G⁻`` — псевдообратная).
2. Ограничения задаются единичными векторами ``C`` (по одному на
   ограниченный примитив). Проектор на допустимое подпространство
   ``P' = P − P C (Cᵀ P C)⁻¹ Cᵀ P``, где ``P = G G⁻`` — проектор на
   подпространство, достижимое декартовыми смещениями.
3. Гессиан проектируется: ``H' = P' H P' + 1000 (1 − P')`` — направления
   ограничений и избыточности получают «бесконечную» жёсткость, шаг в них нулевой.
4. Шаг ``Δq = −H'⁻¹ P' g_q`` плюс восстановление ограничений
   ``Δq_c = P C (Cᵀ P C)⁻¹ (q_target − q_current)``.
5. Обратное преобразование к декартовым итерациями
   ``x ← x + Bᵀ G⁻ (Δq − (q(x) − q₀))``.

Гессиан во внутренних координатах начинается с модели Линдха и уточняется
BFGS по паре ``(Δq, Δg_q)``. Замороженные атомы вводятся как декартовы
примитивы с ограничениями — тем же механизмом, что и остальные ограничения.

Критерии сходимости те же, что в декартовом оптимизаторе, но сила — это
декартов градиент **без** составляющей, уравновешенной ограничениями
(``Bᵀ P' g_q``), а смещение — реальное декартово смещение шага.
Дополнительно требуется выполнение ограничений (невязка < 1e-5 бор/рад).

Состояние для продолжения (:class:`OptimizerState`): ``hessian`` и ``previous_*``
относятся к внутренним координатам; ``coordinates``, ``gradient`` и
``displacement`` — декартовы. Набор примитивов восстанавливается по исходной
геометрии задания, поэтому в контрольную точку не пишется.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.internals import (
    InternalSystem,
    Primitive,
    build_primitives,
    canonical_key,
    lindh_hessian,
)
from quantumlab.engine.optimizer import (
    Constraint,
    EnergyAndGradient,
    OptimizationResult,
    OptimizationSettings,
    OptimizationStep,
    OptimizerState,
    StateSink,
    _bfgs_update,
    _flatten,
    _norms,
    _unflatten,
)

Array = npt.NDArray[np.float64]

#: Жёсткость, назначаемая проектором направлениям ограничений и избыточности.
_PROJECTED_STIFFNESS = 1000.0

#: Допустимая невязка ограничения при проверке сходимости (бор или радианы).
CONSTRAINT_TOLERANCE = 1e-5

#: Выше этой невязки шаг принимается без проверки понижения энергии: восстановление
#: ограничения из нарушенной стартовой геометрии законно повышает энергию.
_RESTORE_WITHOUT_ENERGY_CHECK = 1e-3

_MAX_STEP_HALVINGS = 5

#: Допуск на рост энергии при приёме шага (хартри): ниже уровня шума SCF рост
#: не означает, что шаг плох, а отказ от него останавливал оптимизацию вблизи минимума.
_ENERGY_NOISE = 1e-8
_LINEAR_LIMIT = math.radians(175.0)


def _constraint_primitive(constraint: Constraint) -> Primitive:
    kind = {2: "stretch", 3: "bend", 4: "torsion"}.get(len(constraint.atoms))
    if kind is None:
        msg = "Ограничение задаётся 2, 3 или 4 атомами"
        raise ValueError(msg)
    return Primitive(kind, tuple(constraint.atoms))


def _validate_constraints(molecule: Molecule, constraints: tuple[Constraint, ...]) -> None:
    for constraint in constraints:
        if len(constraint.atoms) not in (2, 3, 4):
            msg = "Ограничение задаётся 2, 3 или 4 атомами"
            raise ValueError(msg)
        if len(set(constraint.atoms)) != len(constraint.atoms):
            msg = f"В ограничении повторяются атомы: {constraint.atoms}"
            raise ValueError(msg)
        for atom in constraint.atoms:
            if not 0 <= atom < molecule.n_atoms:
                msg = f"Атом {atom} в ограничении не существует (всего {molecule.n_atoms})"
                raise IndexError(msg)
        if (
            len(constraint.atoms) == 3
            and constraint.value is not None
            and not 0.0 < constraint.value < _LINEAR_LIMIT
        ):
            msg = "Ограничение угла допустимо в интервале (0°, 175°)"
            raise ValueError(msg)
        if len(constraint.atoms) == 2 and constraint.value is not None and constraint.value <= 0:
            msg = "Длина связи в ограничении должна быть положительной"
            raise ValueError(msg)


class _ConstraintSet:
    """Ограниченные примитивы: индексы, целевые значения и построение проекторов."""

    def __init__(self, indices: list[int], targets: Array, system: InternalSystem) -> None:
        self.indices = indices
        self.targets = targets
        self.system = system

    @property
    def active(self) -> bool:
        return bool(self.indices)

    def vectors(self) -> Array:
        """Матрица ``C`` (``n_q × n_c``) из единичных столбцов."""
        matrix = np.zeros((self.system.size, len(self.indices)))
        for column, index in enumerate(self.indices):
            matrix[index, column] = 1.0
        return matrix

    def violation(self, q: Array) -> Array:
        """Невязка ограничений ``q_target − q_current`` с учётом периодичности."""
        if not self.indices:
            return np.zeros(0)
        current = q[self.indices]
        delta = self.targets - current
        for position, index in enumerate(self.indices):
            if self.system.primitives[index].is_periodic:
                delta[position] = (delta[position] + math.pi) % (2 * math.pi) - math.pi
        return np.asarray(delta)

    def projector(self, projector: Array) -> tuple[Array, Array | None]:
        """Проектор ``P'`` и матрица ``P C (Cᵀ P C)⁻¹`` (``None`` без ограничений)."""
        if not self.indices:
            return projector, None
        c = self.vectors()
        pc = projector @ c
        gram = c.T @ pc
        # Ограничения могут быть линейно зависимы в пространстве допустимых
        # смещений (например, угол и две длины замкнутого треугольника);
        # псевдообратная даёт согласованное решение вместо сингулярности.
        inverse = np.linalg.pinv(gram, rcond=1e-10)
        correction = pc @ inverse
        return projector - correction @ pc.T, correction


def _build_constraints(
    molecule: Molecule, options: OptimizationSettings
) -> tuple[InternalSystem, _ConstraintSet]:
    _validate_constraints(molecule, options.constraints)
    frozen = tuple(sorted(set(options.frozen_atoms)))
    for index in frozen:
        if not 0 <= index < molecule.n_atoms:
            msg = f"Замороженный атом с индексом {index} не существует (всего {molecule.n_atoms})"
            raise IndexError(msg)

    constrained = tuple(_constraint_primitive(c) for c in options.constraints)
    system = build_primitives(molecule, cartesian_atoms=frozen, extra=constrained)
    x0 = _flatten(molecule)
    q0 = system.values(x0)

    lookup = {canonical_key(p): i for i, p in enumerate(system.primitives)}
    indices: list[int] = []
    targets: list[float] = []
    for constraint, primitive in zip(options.constraints, constrained, strict=True):
        index = lookup[canonical_key(primitive)]
        if index in indices:
            msg = f"Координата {constraint.atoms} ограничена дважды"
            raise ValueError(msg)
        indices.append(index)
        targets.append(q0[index] if constraint.value is None else float(constraint.value))
    for atom in frozen:
        for axis in range(3):
            index = lookup[canonical_key(Primitive("cartesian", (atom,), axis))]
            indices.append(index)
            targets.append(q0[index])
    return system, _ConstraintSet(indices, np.asarray(targets, dtype=float), system)


def _solve(hessian: Array, gradient: Array, trust: float) -> Array:
    """Шаг ``−H⁻¹g`` с откатом на спуск при неположительной определённости."""
    try:
        cholesky = np.linalg.cholesky(hessian)
        step = -np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, gradient))
    except np.linalg.LinAlgError:
        norm = float(np.linalg.norm(gradient))
        step = -gradient * (trust / norm) if norm > 0 else np.zeros_like(gradient)
    return np.asarray(step)


def optimize_redundant(
    molecule: Molecule,
    energy_and_gradient: EnergyAndGradient,
    options: OptimizationSettings,
    *,
    resume: OptimizerState | None = None,
    on_state: StateSink | None = None,
) -> OptimizationResult:
    """Оптимизация в избыточных внутренних координатах; контракт как у :func:`optimize_geometry`."""
    system, constraints = _build_constraints(molecule, options)
    x = _flatten(molecule)
    model_hessian = lindh_hessian(molecule, system, x)
    hessian = model_hessian.copy()

    structure = molecule
    history: list[OptimizationStep] = []
    previous_step: Array | None = None
    previous_gradient: Array | None = None
    displacement = np.zeros(3 * molecule.n_atoms)
    first_index = 0
    rejections = 0

    if resume is not None:
        if resume.coordinates.shape != x.shape or resume.hessian.shape != hessian.shape:
            msg = "Состояние оптимизатора не соответствует набору внутренних координат"
            raise ValueError(msg)
        if resume.previous_step is not None and resume.previous_step.shape != (system.size,):
            msg = "Шаг в контрольной точке не соответствует набору внутренних координат"
            raise ValueError(msg)
        if resume.previous_gradient is not None and resume.previous_gradient.shape != (
            system.size,
        ):
            msg = "Градиент в контрольной точке не соответствует набору внутренних координат"
            raise ValueError(msg)
        x = resume.coordinates.copy()
        structure = _unflatten(molecule, x)
        hessian = resume.hessian.copy()
        previous_step = None if resume.previous_step is None else resume.previous_step.copy()
        previous_gradient = (
            None if resume.previous_gradient is None else resume.previous_gradient.copy()
        )
        displacement = resume.displacement.copy()
        history = list(resume.history)
        first_index = resume.step_index
        energy = resume.energy_hartree
        gradient = resume.gradient.reshape(molecule.n_atoms, 3).copy()
    else:
        energy, gradient = energy_and_gradient(structure)

    def analyse(point: Array, cart_gradient: Array) -> tuple[Array, Array, Array, Array, Array]:
        """``(g_q, g'_q, P', P, силы в декартовых без реакций связей)``."""
        wilson = system.wilson(point)
        inverse, projector = system.pseudo_inverse_of_gram(wilson)
        g_q = inverse @ (wilson @ cart_gradient.reshape(-1))
        constrained_projector, _ = constraints.projector(projector)
        g_projected = constrained_projector @ g_q
        force = wilson.T @ g_projected
        return g_q, g_projected, constrained_projector, projector, force

    for index in range(first_index, options.max_steps + 1):
        g_q, g_projected, constrained_projector, projector, force = analyse(x, gradient)
        max_force, rms_force = _norms(force)
        if index == 0:
            max_step: float | None = None
            rms_step: float | None = None
        else:
            max_step, rms_step = _norms(displacement)
        if resume is None or index != first_index:
            history.append(
                OptimizationStep(
                    index=index,
                    energy_hartree=energy,
                    max_force=max_force,
                    rms_force=rms_force,
                    max_displacement=max_step,
                    rms_displacement=rms_step,
                )
            )
        if on_state is not None:
            on_state(
                OptimizerState(
                    step_index=index,
                    coordinates=x.copy(),
                    energy_hartree=energy,
                    gradient=gradient.reshape(-1).copy(),
                    hessian=hessian.copy(),
                    previous_step=None if previous_step is None else previous_step.copy(),
                    previous_gradient=(
                        None if previous_gradient is None else previous_gradient.copy()
                    ),
                    displacement=displacement.copy(),
                    history=tuple(history),
                )
            )

        q = system.values(x)
        violation = constraints.violation(q)
        violation_size = float(np.max(np.abs(violation))) if violation.size else 0.0
        force_converged = max_force < options.max_force and rms_force < options.rms_force
        step_converged = (
            max_step is not None
            and rms_step is not None
            and max_step < options.max_displacement
            and rms_step < options.rms_displacement
        )
        if force_converged and step_converged and violation_size < CONSTRAINT_TOLERANCE:
            return OptimizationResult(
                molecule=structure,
                energy_hartree=energy,
                gradient=force.reshape(molecule.n_atoms, 3),
                converged=True,
                steps=index,
                history=tuple(history),
                reason_key="optimization.converged",
            )
        if index == options.max_steps:
            break

        if previous_step is not None and previous_gradient is not None:
            hessian = _bfgs_update(hessian, previous_step, g_q - previous_gradient)

        n_q = system.size
        projected_hessian = (
            constrained_projector @ hessian @ constrained_projector
            + _PROJECTED_STIFFNESS * (np.eye(n_q) - constrained_projector)
        )
        step_q = _solve(projected_hessian, g_projected, options.trust_radius)
        _, correction = constraints.projector(projector)
        if correction is not None:
            step_q = step_q + correction @ violation
        largest = float(np.max(np.abs(step_q)))
        if largest > options.trust_radius:
            step_q *= options.trust_radius / largest

        gradient_before = g_q
        new_x, new_energy, new_gradient, accepted_step = x, energy, gradient, np.zeros(n_q)
        accepted = False
        factor = 1.0
        for _ in range(_MAX_STEP_HALVINGS):
            candidate_x, _error = system.back_transform(x, factor * step_q)
            candidate = _unflatten(molecule, candidate_x)
            candidate_energy, candidate_gradient = energy_and_gradient(candidate)
            if (
                candidate_energy <= energy + _ENERGY_NOISE
                or violation_size > _RESTORE_WITHOUT_ENERGY_CHECK
            ):
                new_x, new_energy, new_gradient = candidate_x, candidate_energy, candidate_gradient
                accepted_step = system.difference(system.values(candidate_x), q)
                structure = candidate
                accepted = True
                break
            factor /= 2.0
        if accepted:
            rejections = 0
            displacement = new_x - x
            previous_step = accepted_step
            previous_gradient = gradient_before
        else:
            # Энергия растёт даже на укороченном шаге: остаёмся в точке и
            # возвращаемся к модельному гессиану — накопленный BFGS-гессиан
            # здесь явно вводит в заблуждение.
            rejections += 1
            if rejections >= 2:
                # Даже модельный гессиан не даёт понижения энергии: продолжать
                # бессмысленно, а тратить оставшиеся шаги на повтор того же — нечестно.
                return OptimizationResult(
                    molecule=structure,
                    energy_hartree=energy,
                    gradient=force.reshape(molecule.n_atoms, 3),
                    converged=False,
                    steps=index,
                    history=tuple(history),
                    reason_key="optimization.stalled",
                )
            displacement = np.zeros_like(x)
            previous_step = None
            previous_gradient = None
            hessian = model_hessian.copy()
        x, energy, gradient = new_x, new_energy, new_gradient

    return OptimizationResult(
        molecule=structure,
        energy_hartree=energy,
        gradient=analyse(x, gradient)[4].reshape(molecule.n_atoms, 3),
        converged=False,
        steps=options.max_steps,
        history=tuple(history),
        reason_key="optimization.max_steps_reached",
    )
