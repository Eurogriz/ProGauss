"""Оптимизация геометрии: декартов квазиньютоновский метод с BFGS (§11 ТЗ).

Алгоритм
--------
Квазиньютоновский метод: шаг ``Δx = −H⁻¹g``, где ``H`` — приближение гессиана,
обновляемое формулой BFGS по паре ``(Δx, Δg)``. Гессиан не вычисляется и не
хранится в обращённом виде: BFGS даёт сверхлинейную сходимость вблизи минимума
при стоимости одного градиента на шаг.

Почему именно так
-----------------
* **Сложность.** Шаг — решение СЛАУ ``O(n³)`` при ``n = 3N``; градиент
  доминирует. Точный гессиан стоил бы в разы дороже и требовал вторых
  производных интегралов.
* **Где деградирует.** BFGS плохо ведёт себя вдали от минимума и на
  седловых точках; плотный ``H`` не масштабируется на тысячи атомов
  (нужны L-BFGS или внутренние координаты).
* **Внутренние координаты.** Избыточные внутренние координаты и ограничения
  реализованы отдельно (:mod:`quantumlab.engine.redundant_optimizer`);
  этот модуль остаётся декартовым методом и диспетчером.

Надёжность
----------
* **Trust radius.** Если ``‖Δx‖`` превышает доверительный радиус, шаг
  укорачивается до него: линейная модель за большим шагом неверна.
* **Положительная определённость.** Если ``H`` перестаёт быть положительно
  определённой (BFGS-обновление может это испортить на плохом шаге),
  выполняется откат к методу наискорейшего спуска в пределах доверительного
  радиуса. Молча идти в гору нельзя.
* **Замороженные атомы** исключаются из шага: их координаты не меняются,
  соответствующие строки и столбцы ``H`` не участвуют в решении.

Единицы: координаты внутри — бор, градиент — хартри/бор. Пороги сходимости
приходят из :class:`~quantumlab.domain.spec.OptimizationSpec` в тех же единицах.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.engine.constants import BOHR_TO_ANGSTROM, angstrom_to_bohr

Array = npt.NDArray[np.float64]

EnergyAndGradient = Callable[[Molecule], tuple[float, Array]]

#: Ниже этой нормы шага BFGS-обновление не применяется: при вырожденном шаге
#: оно вносит в гессиан шум, а не информацию.
_MIN_STEP_NORM = 1e-10

#: Сколько раз шаг укорачивается вдвое в поисках понижения энергии.
_MAX_STEP_HALVINGS = 5


@dataclass(frozen=True)
class Constraint:
    """Ограничение на внутреннюю координату (индексы атомов — с нуля).

    ``atoms``: 2 — длина связи (бор), 3 — угол (радианы), 4 — двугранный угол
    (радианы). ``value`` — целевое значение в тех же единицах; ``None`` —
    удерживать то значение, которое координата имеет в исходной геометрии.
    """

    atoms: tuple[int, ...]
    value: float | None = None


@dataclass(frozen=True)
class OptimizationSettings:
    """Параметры оптимизации геометрии.

    ``coordinates``: ``cartesian`` — квазиньютоновский метод в декартовых
    координатах (этот модуль); ``redundant_internal`` — избыточные внутренние
    координаты с проекцией ограничений
    (:mod:`quantumlab.engine.redundant_optimizer`). ``constraints`` допустимы
    только во втором случае.
    """

    max_steps: int = 100
    max_force: float = 0.00045
    rms_force: float = 0.0003
    max_displacement: float = 0.0018
    rms_displacement: float = 0.0012
    trust_radius: float = 0.3
    frozen_atoms: tuple[int, ...] = ()
    coordinates: str = "cartesian"
    constraints: tuple[Constraint, ...] = ()


@dataclass(frozen=True)
class OptimizationStep:
    """Одна итерация оптимизации — для журнала и графика сходимости."""

    index: int
    energy_hartree: float
    max_force: float
    rms_force: float
    max_displacement: float | None
    rms_displacement: float | None


@dataclass(frozen=True)
class OptimizerState:
    """Состояние оптимизатора, достаточное, чтобы продолжить без потери истории.

    Снимок делается в точке цикла **до** BFGS-обновления и выбора шага, поэтому
    продолженный расчёт повторяет те же операции в том же порядке и даёт ту же
    траекторию, что и непрерывный (проверяется тестом бит-в-бит по геометрии).

    Attributes:
        step_index: номер итерации, для которой уже записана строка истории.
        coordinates: координаты всех атомов, бор, плоский вектор.
        energy_hartree: энергия в этой точке.
        gradient: градиент в этой точке, хартри/бор, плоский вектор.
        hessian: текущее приближение гессиана в подпространстве подвижных
            степеней свободы.
        previous_step: шаг, приведший в эту точку (в подпространстве подвижных
            степеней свободы), либо ``None`` на нулевой итерации.
        previous_gradient: градиент до этого шага, либо ``None``.
        displacement: полный вектор последнего смещения (для критерия шага).
        history: журнал итераций, включая текущую.
    """

    step_index: int
    coordinates: Array
    energy_hartree: float
    gradient: Array
    hessian: Array
    previous_step: Array | None
    previous_gradient: Array | None
    displacement: Array
    history: tuple[OptimizationStep, ...]


#: Приёмник состояний оптимизатора (запись контрольной точки).
StateSink = Callable[[OptimizerState], None]


@dataclass(frozen=True)
class OptimizationResult:
    """Итог оптимизации геометрии."""

    molecule: Molecule
    energy_hartree: float
    gradient: Array
    converged: bool
    steps: int
    history: tuple[OptimizationStep, ...]
    reason_key: str

    @property
    def max_force(self) -> float:
        """Наибольшая компонента силы в конечной точке."""
        return float(np.max(np.abs(self.gradient)))


def _flatten(molecule: Molecule) -> Array:
    """Координаты всех атомов одним вектором, в борах."""
    return np.array([angstrom_to_bohr(value) for atom in molecule.atoms for value in atom.position])


def _unflatten(molecule: Molecule, coordinates: Array) -> Molecule:
    """Собирает молекулу из вектора координат (бор → ангстрем)."""
    atoms = tuple(
        Atom(
            symbol=atom.symbol,
            position=(
                float(coordinates[3 * i] * BOHR_TO_ANGSTROM),
                float(coordinates[3 * i + 1] * BOHR_TO_ANGSTROM),
                float(coordinates[3 * i + 2] * BOHR_TO_ANGSTROM),
            ),
        )
        for i, atom in enumerate(molecule.atoms)
    )
    return molecule.model_copy(update={"atoms": atoms})


def _free_mask(molecule: Molecule, frozen: tuple[int, ...]) -> Array:
    """Булева маска подвижных степеней свободы."""
    mask = np.ones(3 * molecule.n_atoms, dtype=bool)
    for index in frozen:
        mask[3 * index : 3 * index + 3] = False
    return mask


def _norms(values: Array) -> tuple[float, float]:
    """Максимум и среднеквадратичное значение по компонентам."""
    return float(np.max(np.abs(values))), float(np.sqrt(np.mean(np.square(values))))


def _bfgs_update(hessian: Array, step: Array, gradient_change: Array) -> Array:
    """Обновление BFGS: ``H ← H + yyᵗ/(yᵗs) − (Hs)(Hs)ᵗ/(sᵗHs)``."""
    curvature = float(gradient_change @ step)
    if curvature <= 1e-12:
        # Кривизна неположительна: такое обновление разрушило бы положительную
        # определённость. Гессиан оставляем прежним — шаг уже сделан.
        return hessian
    hessian_step = hessian @ step
    denominator = float(step @ hessian_step)
    if denominator <= 1e-12:
        return hessian
    return np.asarray(
        hessian
        + np.outer(gradient_change, gradient_change) / curvature
        - np.outer(hessian_step, hessian_step) / denominator
    )


def optimize_geometry(
    molecule: Molecule,
    energy_and_gradient: EnergyAndGradient,
    settings: OptimizationSettings | None = None,
    *,
    resume: OptimizerState | None = None,
    on_state: StateSink | None = None,
) -> OptimizationResult:
    """Оптимизирует геометрию, возвращая структуру и сведения о сходимости.

    Функция не бросает исключение при исчерпании шагов: несошедшаяся
    оптимизация — это результат, который пользователь обязан увидеть вместе с
    причиной, а не аварийное завершение. Признак ``converged`` и ключ
    ``reason_key`` позволяют интерфейсу объяснить, что именно не сошлось.

    ``on_state`` вызывается на каждой итерации (после записи в журнал и проверки
    сходимости) со снимком :class:`OptimizerState`; ``resume`` продолжает по
    такому снимку, не пересчитывая энергию и градиент в точке остановки.
    ``molecule`` — всегда исходная структура задания: из неё берутся символы
    атомов, заряд и кратность.
    """
    options = settings or OptimizationSettings()
    if options.coordinates == "redundant_internal":
        from quantumlab.engine.redundant_optimizer import optimize_redundant

        return optimize_redundant(
            molecule, energy_and_gradient, options, resume=resume, on_state=on_state
        )
    if options.coordinates != "cartesian":
        msg = f"Система координат {options.coordinates!r} не реализована"
        raise ValueError(msg)
    if options.constraints:
        msg = "Ограничения координат требуют redundant_internal"
        raise ValueError(msg)
    frozen = tuple(options.frozen_atoms)
    for index in frozen:
        if not 0 <= index < molecule.n_atoms:
            msg = f"Замороженный атом с индексом {index} не существует (всего {molecule.n_atoms})"
            raise IndexError(msg)

    free = _free_mask(molecule, frozen)
    coordinates = _flatten(molecule)
    n_free = int(np.count_nonzero(free))
    hessian = np.eye(n_free)

    structure = molecule
    history: list[OptimizationStep] = []
    previous_gradient: Array | None = None
    # BFGS требует пару (Δx, Δg) с ОДНОГО шага. Шаг, сделанный на итерации k−1,
    # должен сочетаться с изменением градиента между k−1 и k — перепутать их
    # значит обновить гессиан бессмысленной парой и потерять сходимость.
    previous_step: Array | None = None
    displacement = np.zeros(3 * molecule.n_atoms)
    first_index = 0

    if resume is not None:
        if resume.coordinates.shape != coordinates.shape or resume.hessian.shape != hessian.shape:
            msg = "Состояние оптимизатора не соответствует размеру задачи"
            raise ValueError(msg)
        if resume.previous_step is not None and resume.previous_step.shape != (n_free,):
            msg = "Шаг в контрольной точке не соответствует размеру задачи"
            raise ValueError(msg)
        if resume.previous_gradient is not None and resume.previous_gradient.shape != (
            coordinates.shape
        ):
            msg = "Градиент в контрольной точке не соответствует размеру задачи"
            raise ValueError(msg)
        coordinates = resume.coordinates.copy()
        structure = _unflatten(molecule, coordinates)
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

    for index in range(first_index, options.max_steps + 1):
        flat_gradient = gradient.reshape(-1)
        max_force, rms_force = _norms(flat_gradient)
        if index == 0:
            max_step: float | None = None
            rms_step: float | None = None
        else:
            max_step, rms_step = _norms(displacement)
        # При продолжении строка текущей итерации уже есть в журнале.
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
                    coordinates=coordinates.copy(),
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

        force_converged = max_force < options.max_force and rms_force < options.rms_force
        step_converged = (
            max_step is not None
            and rms_step is not None
            and max_step < options.max_displacement
            and rms_step < options.rms_displacement
        )
        if force_converged and step_converged:
            return OptimizationResult(
                molecule=structure,
                energy_hartree=energy,
                gradient=gradient,
                converged=True,
                steps=index,
                history=tuple(history),
                reason_key="optimization.converged",
            )
        if index == options.max_steps:
            break

        # BFGS-обновление: шаг, уже сделанный на прошлой итерации, и изменение
        # градиента за тот же шаг.
        if previous_step is not None and previous_gradient is not None:
            hessian = _bfgs_update(
                hessian, previous_step, (flat_gradient - previous_gradient)[free]
            )

        # Шаг в подпространстве подвижных степеней свободы.
        step_free = _solve_step(hessian, flat_gradient[free], options.trust_radius)
        step = np.zeros_like(coordinates)
        step[free] = step_free

        # Градиент ДО шага запоминается раньше, чем ``_accept_step`` вернёт новый:
        # прежняя версия сохраняла уже новый градиент, из-за чего изменение
        # градиента между итерациями всегда было нулевым, кривизна — нулевой, и
        # BFGS-обновление молча пропускалось. Оптимизатор вырождался в спуск с
        # единичным гессианом.
        gradient_before_step = flat_gradient.copy()
        energy, gradient, coordinates, step, structure = _accept_step(
            energy, gradient, coordinates, step, molecule, energy_and_gradient
        )
        displacement = step
        previous_step = step[free]
        previous_gradient = gradient_before_step

    return OptimizationResult(
        molecule=structure,
        energy_hartree=energy,
        gradient=gradient,
        converged=False,
        steps=options.max_steps,
        history=tuple(history),
        reason_key="optimization.max_steps_reached",
    )


def _accept_step(
    energy: float,
    gradient: Array,
    coordinates: Array,
    step: Array,
    molecule: Molecule,
    energy_and_gradient: EnergyAndGradient,
) -> tuple[float, Array, Array, Array, Molecule]:
    """Делает шаг, укорачивая его вдвое, пока энергия не перестанет расти.

    Квазиньютоновская модель за большим шагом неверна, особенно в декартовых
    координатах: без укорачивания оптимизация уходит вверх по энергии там, где
    начальное приближение гессиана плохое.

    Возвращает согласованный набор ``(энергия, градиент, координаты, шаг,
    структура)``: все пять величин обязаны относиться к одной точке. Отдавать
    градиент из отвергнутой точки вместе с исходными координатами — значит
    сломать и BFGS-обновление, и проверку сходимости.
    """
    current = step
    for _ in range(_MAX_STEP_HALVINGS):
        next_coordinates = coordinates + current
        candidate = _unflatten(molecule, next_coordinates)
        new_energy, new_gradient = energy_and_gradient(candidate)
        if new_energy <= energy:
            return new_energy, new_gradient, next_coordinates, current, candidate
        current = current / 2.0
    # Энергия растёт даже на укороченном шаге. Остаёмся в прежней точке с
    # прежним градиентом и нулевым шагом: следующая итерация увидит реальное
    # состояние, а не смесь двух точек.
    return energy, gradient, coordinates, np.zeros_like(step), molecule


def _solve_step(hessian: Array, gradient: Array, trust_radius: float) -> Array:
    """Шаг ``−H⁻¹g``, укороченный до доверительного радиуса.

    Если ``H`` не положительно определена, решение не имеет смысла минимума —
    в этом случае выполняется шаг наискорейшего спуска той же допустимой
    длины. Откат явный: он попадает в журнал, а не прячется.
    """
    try:
        cholesky = np.linalg.cholesky(hessian)
        step = -np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, gradient))
    except np.linalg.LinAlgError:
        norm = float(np.linalg.norm(gradient))
        step = -gradient * (trust_radius / norm) if norm > 0 else np.zeros_like(gradient)

    length = float(np.linalg.norm(step))
    if length > trust_radius:
        step *= trust_radius / length
    if float(np.linalg.norm(step)) < _MIN_STEP_NORM:
        return np.zeros_like(step)
    return np.asarray(step)
