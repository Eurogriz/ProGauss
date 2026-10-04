"""Избыточные внутренние координаты: примитивы, матрица Вильсона, модельный гессиан.

Что это
-------
Набор примитивных внутренних координат — длины связей, валентные и линейные
углы, двугранные углы (и декартовы компоненты замороженных атомов) — строится
по связности исходной геометрии. Примитивов больше, чем ``3N − 6``: они
**избыточны**, и матрица Вильсона ``B = ∂q/∂x`` (размер ``n_q × 3N``) имеет
ранг ``3N − 6``. Обращение делается через псевдообратную матрицу
``G⁻ = (B Bᵀ)⁻`` с отсечением нулевых собственных значений.

Сходимость в таких координатах обычно в разы быстрее, чем в декартовых:
гессиан почти диагонален, а нелинейность связей и углов умеренная.

Соглашения
----------
* Декартовы координаты — в борах, длины — в борах, углы — в радианах.
* Двугранный угол определяется на ``(−π, π]``; разности координат всегда
  приводятся к этому интервалу (:meth:`InternalSystem.difference`).
* Линейные углы ``a–b–c`` (θ > 175°) описываются парой линейных изгибов
  ``w·(u + v)`` во взаимно перпендикулярных плоскостях; направления ``w``
  фиксируются по исходной геометрии, поэтому набор координат
  детерминирован — это нужно для продолжения расчёта с контрольной точки.

Ограничение
-----------
Набор примитивов строится один раз по исходной геометрии и не пересматривается:
если в ходе оптимизации рвётся связь или угол проходит через 180°, набор
остаётся прежним (для линейных молекул это учтено линейными изгибами).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.constants import angstrom_to_bohr

Array = npt.NDArray[np.float64]

#: Связь, если расстояние не превышает этот множитель суммы ковалентных радиусов.
BOND_TOLERANCE = 1.3

#: Углы больше этого значения (в градусах) считаются линейными.
LINEAR_ANGLE_DEGREES = 175.0

#: Отсечение собственных значений ``B Bᵀ`` при псевдообращении.
_G_CUTOFF = 1e-7

#: Минимальный синус угла в знаменателях производных (защита от деления на 0).
_MIN_SINE = 1e-8

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Primitive:
    """Одна примитивная внутренняя координата.

    ``kind``: ``stretch``, ``bend``, ``linear_bend``, ``torsion`` или
    ``cartesian``. ``atoms`` — индексы атомов; ``axis`` — номер компоненты
    для ``cartesian``; ``direction`` — фиксированный перпендикулярный вектор
    для ``linear_bend``.
    """

    kind: str
    atoms: tuple[int, ...]
    axis: int = 0
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_periodic(self) -> bool:
        """Периодична ли координата (двугранный угол)."""
        return self.kind == "torsion"


def _unit(vector: Array) -> tuple[Array, float]:
    length = float(np.linalg.norm(vector))
    return vector / length, length


def _angle_and_gradient(a: Array, b: Array, c: Array) -> tuple[float, Array]:
    """Угол ``a–b–c`` и его градиент (3×3: по a, b, c)."""
    u, ru = _unit(a - b)
    v, rv = _unit(c - b)
    cosine = float(np.clip(u @ v, -1.0, 1.0))
    theta = math.acos(cosine)
    sine = max(math.sin(theta), _MIN_SINE)
    d_a = (cosine * u - v) / (sine * ru)
    d_c = (cosine * v - u) / (sine * rv)
    return theta, np.array([d_a, -(d_a + d_c), d_c])


def _torsion_and_gradient(a: Array, b: Array, c: Array, d: Array) -> tuple[float, Array]:
    """Двугранный угол ``a–b–c–d`` и его градиент (4×3), формулы Блонделя—Карплюса."""
    f = a - b
    g = b - c
    h = d - c
    cross_a = np.cross(f, g)
    cross_b = np.cross(h, g)
    norm_a2 = float(cross_a @ cross_a)
    norm_b2 = float(cross_b @ cross_b)
    length_g = float(np.linalg.norm(g))
    sine_part = float(np.cross(cross_b, cross_a) @ g) / length_g
    cosine_part = float(cross_a @ cross_b)
    phi = math.atan2(sine_part, cosine_part)
    fg = float(f @ g)
    hg = float(h @ g)
    grad_a = -length_g / norm_a2 * cross_a
    grad_d = length_g / norm_b2 * cross_b
    grad_b = (
        length_g / norm_a2 * cross_a
        + fg / (norm_a2 * length_g) * cross_a
        - hg / (norm_b2 * length_g) * cross_b
    )
    grad_c = (
        -length_g / norm_b2 * cross_b
        - fg / (norm_a2 * length_g) * cross_a
        + hg / (norm_b2 * length_g) * cross_b
    )
    return phi, np.array([grad_a, grad_b, grad_c, grad_d])


def _wrap(value: float) -> float:
    """Приводит угол к интервалу ``(−π, π]``."""
    wrapped = (value + math.pi) % _TWO_PI - math.pi
    return math.pi if wrapped <= -math.pi else wrapped


def _evaluate(primitive: Primitive, x: Array) -> tuple[float, Array, tuple[int, ...]]:
    """Значение примитива, его градиент по координатам его атомов и индексы атомов."""
    atoms = primitive.atoms
    points = [x[3 * i : 3 * i + 3] for i in atoms]
    if primitive.kind == "stretch":
        unit, length = _unit(points[0] - points[1])
        return length, np.array([unit, -unit]), atoms
    if primitive.kind == "bend":
        theta, grad = _angle_and_gradient(*points)
        return theta, grad, atoms
    if primitive.kind == "torsion":
        phi, grad = _torsion_and_gradient(*points)
        return phi, grad, atoms
    if primitive.kind == "linear_bend":
        w = np.asarray(primitive.direction)
        u, ru = _unit(points[0] - points[1])
        v, rv = _unit(points[2] - points[1])
        d_a = (w - (w @ u) * u) / ru
        d_c = (w - (w @ v) * v) / rv
        return float(w @ (u + v)), np.array([d_a, -(d_a + d_c), d_c]), atoms
    if primitive.kind == "cartesian":
        grad = np.zeros((1, 3))
        grad[0, primitive.axis] = 1.0
        return float(points[0][primitive.axis]), grad, atoms
    msg = f"Неизвестный вид примитива: {primitive.kind}"
    raise ValueError(msg)


class InternalSystem:
    """Набор примитивов с расчётом значений, матрицы Вильсона и обратного преобразования."""

    def __init__(self, primitives: tuple[Primitive, ...], n_atoms: int) -> None:
        """Сохраняет набор примитивов и число атомов."""
        self.primitives = primitives
        self.n_atoms = n_atoms

    @property
    def size(self) -> int:
        """Число примитивов."""
        return len(self.primitives)

    def values(self, x: Array) -> Array:
        """Значения всех примитивов в точке ``x`` (плоский вектор, бор)."""
        return np.array([_evaluate(p, x)[0] for p in self.primitives])

    def wilson(self, x: Array) -> Array:
        """Матрица Вильсона ``B = ∂q/∂x`` (``n_q × 3N``)."""
        matrix = np.zeros((self.size, 3 * self.n_atoms))
        for row, primitive in enumerate(self.primitives):
            _, grad, atoms = _evaluate(primitive, x)
            for local, atom in enumerate(atoms):
                matrix[row, 3 * atom : 3 * atom + 3] = grad[local]
        return matrix

    def difference(self, new: Array, old: Array) -> Array:
        """``q_new − q_old`` с приведением периодических координат к ``(−π, π]``."""
        delta = np.asarray(new - old, dtype=float).copy()
        for index, primitive in enumerate(self.primitives):
            if primitive.is_periodic:
                delta[index] = _wrap(float(delta[index]))
        return delta

    @staticmethod
    def pseudo_inverse_of_gram(wilson: Array) -> tuple[Array, Array]:
        """``G⁻`` и проектор ``P = G G⁻`` на допустимое подпространство.

        ``G = B Bᵀ`` симметрична неотрицательно определена; собственные значения
        ниже порога соответствуют избыточности и отбрасываются.
        """
        gram = wilson @ wilson.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        keep = eigenvalues > _G_CUTOFF
        vectors = eigenvectors[:, keep]
        inverse = (vectors / eigenvalues[keep]) @ vectors.T
        projector = vectors @ vectors.T
        return inverse, projector

    def rank(self, x: Array) -> int:
        """Ранг матрицы Вильсона (число независимых внутренних координат)."""
        singular = np.linalg.svd(self.wilson(x), compute_uv=False)
        return int(np.count_nonzero(singular > math.sqrt(_G_CUTOFF)))

    def back_transform(
        self,
        x_start: Array,
        delta_q: Array,
        *,
        max_iterations: int = 40,
        tolerance: float = 1e-9,
    ) -> tuple[Array, float]:
        """Итеративный переход от смещения во внутренних координатах к декартовым.

        Решает ``q(x) − q(x_start) = Δq`` повторением
        ``x ← x + Bᵀ G⁻ (Δq − (q(x) − q(x_start)))``: линейное приближение
        повторяется, пока невязка не исчезнет. Возвращает лучшую найденную
        точку и невязку (если сходимости нет, это лучшая из попыток, а не
        последняя).
        """
        q_start = self.values(x_start)
        x = x_start.copy()
        best_x = x.copy()
        best_error = float("inf")
        for _ in range(max_iterations):
            actual = self.difference(self.values(x), q_start)
            residual = delta_q - actual
            for index, primitive in enumerate(self.primitives):
                if primitive.is_periodic:
                    residual[index] = _wrap(float(residual[index]))
            error = float(np.max(np.abs(residual)))
            if error < best_error:
                best_error = error
                best_x = x.copy()
            if error < tolerance:
                break
            wilson = self.wilson(x)
            inverse, _ = self.pseudo_inverse_of_gram(wilson)
            x = x + wilson.T @ (inverse @ residual)
        return best_x, best_error


def _covalent_bohr(molecule: Molecule, index: int) -> float:
    return angstrom_to_bohr(molecule.atoms[index].element.covalent_radius)


def _connectivity(molecule: Molecule, x: Array) -> list[tuple[int, int]]:
    """Связи: по ковалентным радиусам плюс кратчайшие связи между фрагментами."""
    n = molecule.n_atoms
    positions = x.reshape(n, 3)
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    bonds = [
        (i, j)
        for i, j in combinations(range(n), 2)
        if distance[i, j]
        < BOND_TOLERANCE * (_covalent_bohr(molecule, i) + _covalent_bohr(molecule, j))
    ]
    # Несвязные фрагменты соединяются кратчайшей парой: иначе их взаимная
    # ориентация не описана ни одной координатой.
    while True:
        component = list(range(n))

        def root(i: int, parent: list[int] = component) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in bonds:
            component[root(i)] = root(j)
        groups = {root(i) for i in range(n)}
        if len(groups) <= 1:
            return bonds
        best: tuple[float, int, int] | None = None
        for i, j in combinations(range(n), 2):
            if root(i) != root(j) and (best is None or distance[i, j] < best[0]):
                best = (float(distance[i, j]), i, j)
        assert best is not None
        bonds.append((best[1], best[2]))


def _perpendicular_pair(axis: Array) -> tuple[Array, Array]:
    """Два единичных вектора, перпендикулярных ``axis`` и друг другу."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second / np.linalg.norm(second)


def build_primitives(
    molecule: Molecule,
    *,
    cartesian_atoms: tuple[int, ...] = (),
    extra: tuple[Primitive, ...] = (),
) -> InternalSystem:
    """Строит избыточный набор примитивов по исходной геометрии.

    ``cartesian_atoms`` — атомы, декартовы компоненты которых добавляются
    (для замораживания); ``extra`` — обязательные примитивы (координаты,
    на которые наложены ограничения): они добавляются, если их ещё нет.

    Выбрасывает :class:`ValueError`, если набор не покрывает ``3N − 6``
    степеней свободы даже после добавления всех попарных расстояний.
    """
    n = molecule.n_atoms
    x = np.array([angstrom_to_bohr(value) for atom in molecule.atoms for value in atom.position])
    if n < 2:
        return InternalSystem(
            tuple(Primitive("cartesian", (i,), axis) for i in range(n) for axis in range(3)), n
        )
    positions = x.reshape(n, 3)
    bonds = _connectivity(molecule, x)
    neighbours: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)

    primitives: list[Primitive] = [Primitive("stretch", (min(b), max(b))) for b in bonds]
    linear_centres: set[tuple[int, int, int]] = set()
    cosine_limit = math.cos(math.radians(LINEAR_ANGLE_DEGREES))

    def is_linear(i: int, j: int, k: int) -> bool:
        u, _ = _unit(positions[i] - positions[j])
        v, _ = _unit(positions[k] - positions[j])
        return float(u @ v) < cosine_limit

    for j in range(n):
        for i, k in combinations(sorted(neighbours[j]), 2):
            if is_linear(i, j, k):
                linear_centres.add((i, j, k))
                axis, _ = _unit(positions[k] - positions[i])
                for direction in _perpendicular_pair(axis):
                    primitives.append(
                        Primitive(
                            "linear_bend",
                            (i, j, k),
                            direction=(
                                float(direction[0]),
                                float(direction[1]),
                                float(direction[2]),
                            ),
                        )
                    )
            else:
                primitives.append(Primitive("bend", (i, j, k)))

    def linear(a: int, b: int, c: int) -> bool:
        return (min(a, c), b, max(a, c)) in linear_centres

    for j, k in bonds:
        for a in neighbours[j]:
            for d in neighbours[k]:
                if a in (k, d) or d in (j, a):
                    continue
                if linear(a, j, k) or linear(j, k, d):
                    continue
                primitives.append(Primitive("torsion", (a, j, k, d)))

    for index in cartesian_atoms:
        primitives.extend(Primitive("cartesian", (index,), axis) for axis in range(3))
    for primitive in extra:
        if primitive not in primitives and canonical_key(primitive) not in {
            canonical_key(p) for p in primitives
        }:
            primitives.append(primitive)

    system = InternalSystem(tuple(primitives), n)
    needed = 3 * n - (5 if _is_linear_molecule(positions) else 6)
    if system.rank(x) < needed:
        # Дополнение всеми попарными расстояниями: для малых систем это дёшево
        # и всегда даёт полный ранг для нелинейных структур.
        present = {canonical_key(p) for p in primitives}
        for i, j in combinations(range(n), 2):
            candidate = Primitive("stretch", (i, j))
            if canonical_key(candidate) not in present:
                primitives.append(candidate)
        system = InternalSystem(tuple(primitives), n)
        if system.rank(x) < needed:
            msg = (
                "Набор внутренних координат не покрывает все степени свободы "
                f"({system.rank(x)} из {needed}); используйте декартовы координаты"
            )
            raise ValueError(msg)
    return system


def canonical_key(primitive: Primitive) -> tuple[object, ...]:
    """Ключ примитива, не зависящий от направления перечисления атомов."""
    atoms = primitive.atoms
    if primitive.kind in ("stretch", "bend", "torsion", "linear_bend"):
        atoms = min(atoms, atoms[::-1])
    return (primitive.kind, atoms, primitive.axis, primitive.direction)


def _is_linear_molecule(positions: Array) -> bool:
    if len(positions) < 3:
        return True
    centred = positions - positions.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return bool(singular[1] < 1e-6 * max(singular[0], 1.0))


# --------------------------------------------------------------------------- #
# Модельный гессиан Линдха (Lindh, Bernhardsson, Karlström, Malmqvist 1995)
# --------------------------------------------------------------------------- #
_LINDH_ALPHA = np.array(
    [[1.0000, 0.3949, 0.3949], [0.3949, 0.2800, 0.2800], [0.3949, 0.2800, 0.2800]]
)
_LINDH_R_REF = np.array(
    [[1.3500, 2.1000, 2.5300], [2.1000, 2.8700, 3.4000], [2.5300, 3.4000, 3.4000]]
)


def _period(z: int) -> int:
    if z <= 2:
        return 0
    if z <= 10:
        return 1
    return 2


def lindh_hessian(molecule: Molecule, system: InternalSystem, x: Array) -> Array:
    """Диагональный модельный гессиан во внутренних координатах (хартри/бор², /рад²)."""
    n = molecule.n_atoms
    positions = x.reshape(n, 3)
    periods = [_period(atom.element.z) for atom in molecule.atoms]

    def rho(i: int, j: int) -> float:
        r = float(np.linalg.norm(positions[i] - positions[j]))
        a = _LINDH_ALPHA[periods[i], periods[j]]
        r_ref = _LINDH_R_REF[periods[i], periods[j]]
        return math.exp(a * (r_ref**2 - r**2))

    diagonal = np.empty(system.size)
    for index, primitive in enumerate(system.primitives):
        atoms = primitive.atoms
        if primitive.kind == "stretch":
            diagonal[index] = 0.45 * rho(*atoms)
        elif primitive.kind in ("bend", "linear_bend"):
            diagonal[index] = 0.15 * rho(atoms[0], atoms[1]) * rho(atoms[1], atoms[2])
        elif primitive.kind == "torsion":
            diagonal[index] = (
                0.005 * rho(atoms[0], atoms[1]) * rho(atoms[1], atoms[2]) * rho(atoms[2], atoms[3])
            )
        else:
            diagonal[index] = 0.1
    return np.diag(np.maximum(diagonal, 1e-3))
