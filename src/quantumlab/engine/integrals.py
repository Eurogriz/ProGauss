"""Одно- и двухэлектронные интегралы по декартовым гауссианам.

Алгоритм — McMurchie–Davidson (соотношения из Helgaker, Jørgensen, Olsen,
*Molecular Electronic-Structure Theory*, гл. 9): коэффициенты Эрмитовых
разложений ``E`` и эрмитовы кулоновские интегралы ``R`` вычисляются
рекуррентно, что даёт единый код для любого углового момента.

Сложность без скрининга: одноэлектронные — O(N²), двухэлектронные — O(N⁴) по
числу оболочек. Это **референсная** реализация (ADR-002): плотные массивы,
один поток, без отсечек — её задача быть эталоном корректности.

Двухэлектронные интегралы собираются **пачками квартетов одного класса**
(одинаковые угловые моменты и длины сжатий): стоимость здесь определялась
числом обращений к NumPy, а не арифметикой, и объединение квартетов в один
массив ускорило сборку в 20 раз на бензоле/6-31G, не изменив значений
(совпадение с поквартиетной сборкой проверяется тестом по всему тензору).
Поквартиетный путь сохранён как эталон: :func:`_quartet_block` и
:func:`_build_electron_repulsion_quartetwise`.

Единицы — атомные (бор, хартри).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import BasisSet, Shell, cartesian_powers
from quantumlab.engine.constants import PI_5_2, angstrom_to_bohr

#: Ниже этого аргумента функция Бойса считается рядом Тейлора.
_BOYS_SERIES_LIMIT = 6.0

#: Точность обрыва ряда Тейлора для функции Бойса. Число членов подбирается
#: по аргументу (:func:`_series_terms`): фиксированное число либо тратит время
#: на малых ``x``, либо теряет знаки на ``x`` близких к границе ветвей.
_BOYS_SERIES_TOLERANCE = 1e-17

#: Верхняя граница кеша функции Бойса. Достаточно велика, чтобы обычный
#: расчёт не вытеснял свои же значения, и достаточно мала, чтобы не
#: расходовать память на больших системах (~25 МБ).
_BOYS_CACHE_SIZE: Final = 1 << 18

#: Сигнатура примитивного интеграла: (a, la, A, b, lb, B) → значение.
PrimitiveKernel = Callable[
    [
        float,
        tuple[int, int, int],
        tuple[float, float, float],
        float,
        tuple[int, int, int],
        tuple[float, float, float],
    ],
    float,
]

Powers = tuple[int, int, int]
Point = tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Рекурсии
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1 << 16)
def _hermite_coefficient(i: int, j: int, t: int, q: float, a: float, b: float) -> float:
    r"""Коэффициент Эрмитовского разложения ``E^{ij}_t`` для одной оси.

    Определение: для одной декартовой оси

    .. math::

        (x-A)^i (x-B)^j e^{-a(x-A)^2} e^{-b(x-B)^2} = \sum_t E^{ij}_t \Lambda_t,

    где ``\Lambda_t = (\partial/\partial P)^t e^{-p(x-P)^2}`` — эрмитова
    гауссова функция, ``p = a + b``, ``P = (aA + bB)/p``, ``Q = A - B``.

    Рекурсия выводится тождеством ``(x-P)\Lambda_t = \Lambda_{t+1}/(2p) + t\Lambda_{t-1}``
    и соотношениями ``A - P = bQ/p``, ``B - P = -aQ/p``::

        E^{i+1,j}_t = E^{i,j}_{t-1}/(2p) + (t+1) E^{i,j}_{t+1} − (bQ/p) E^{i,j}_t
        E^{i,j+1}_t = E^{i,j}_{t-1}/(2p) + (t+1) E^{i,j}_{t+1} + (aQ/p) E^{i,j}_t
        E^{00}_0    = exp(−μQ²),   μ = ab/p

    Именно в такой форме ``E^{ij}_0`` совпадает с интегральным определением
    ``E^{ij}_0 = √(p/π) ∫ (x-A)^i (x-B)^j e^{-a(x-A)²-b(x-B)²} dx``; более
    распространённая запись через ``1/(2μ)`` и ``±Q`` относится к другой
    нормировке коэффициентов и даёт неверные интегралы (проверено численно).
    """
    if t < 0 or t > i + j:
        return 0.0
    p = a + b
    if i == 0 and j == 0:
        mu = a * b / p
        return math.exp(-mu * q * q) if t == 0 else 0.0
    if i > 0:
        return (
            _hermite_coefficient(i - 1, j, t - 1, q, a, b) / (2.0 * p)
            + (t + 1) * _hermite_coefficient(i - 1, j, t + 1, q, a, b)
            - (b * q / p) * _hermite_coefficient(i - 1, j, t, q, a, b)
        )
    return (
        _hermite_coefficient(i, j - 1, t - 1, q, a, b) / (2.0 * p)
        + (t + 1) * _hermite_coefficient(i, j - 1, t + 1, q, a, b)
        + (a * q / p) * _hermite_coefficient(i, j - 1, t, q, a, b)
    )


@lru_cache(maxsize=_BOYS_CACHE_SIZE)
def _boys(n: int, x: float) -> float:
    """Функция Бойса ``F_n(x) = ∫₀¹ t^{2n} e^{-x t²} dt``.

    Для малых ``x`` — ряд Тейлора (устойчив, пока члены не начинают
    сокращаться), для больших — рекурсия вверх от
    ``F_0 = ½√(π/x)·erf(√x)``, которая устойчива при ``x ≫ n``.

    Результат кешируется. ``x = α·R²_PQ`` определяется одним примитивным
    квартетом и не зависит от угловых моментов, поэтому при сборке
    производных одно и то же значение запрашивается сотни раз: на воде/STO-3G
    было 1.34 млн вычислений ряда при ~30 тыс. различных аргументов.
    Кеш ограничен по размеру, потому что число различных квартетов растёт
    как четвёртая степень размера системы.
    """
    if x < 1e-12:
        return 1.0 / (2 * n + 1)
    if x < _BOYS_SERIES_LIMIT:
        total = 0.0
        term = 1.0
        for k in range(_series_terms(x)):
            if k:
                term *= -x / k
            total += term / (2 * n + 2 * k + 1)
        return total
    value = 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))
    exponent = math.exp(-x)
    for order in range(n):
        value = ((2 * order + 1) * value - exponent) / (2.0 * x)
    return value


def _hermite_coulomb(t: int, u: int, v: int, n: int, p: float, pq: Point, rpc: float) -> float:
    """Эрмитов кулоновский интеграл ``R^n_{tuv}(p, P − Q)``.

    Рекурсия (Helgaker 9.9.15–9.9.18)::

        R^0_{000}  = F_0(p R²)
        R^n_{000}  = (−2p)^n F_n(p R²)
        R^n_{t u v} = PQ_x R^{n+1}_{t−1,u,v} + (t−1) R^{n+1}_{t−2,u,v}   (t > 0)

    и циклически для ``u`` и ``v``. Отрицательные индексы дают ноль.
    """
    if t == 0 and u == 0 and v == 0:
        return (-2.0 * p) ** n * _boys(n, p * rpc * rpc)
    if t > 0:
        second = (t - 1) * _hermite_coulomb(t - 2, u, v, n + 1, p, pq, rpc) if t > 1 else 0.0
        return pq[0] * _hermite_coulomb(t - 1, u, v, n + 1, p, pq, rpc) + second
    if u > 0:
        second = (u - 1) * _hermite_coulomb(t, u - 2, v, n + 1, p, pq, rpc) if u > 1 else 0.0
        return pq[1] * _hermite_coulomb(t, u - 1, v, n + 1, p, pq, rpc) + second
    second = (v - 1) * _hermite_coulomb(t, u, v - 2, n + 1, p, pq, rpc) if v > 1 else 0.0
    return pq[2] * _hermite_coulomb(t, u, v - 1, n + 1, p, pq, rpc) + second


# --------------------------------------------------------------------------- #
# Примитивные интегралы
# --------------------------------------------------------------------------- #
def _hermite_product(
    a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point
) -> float:
    """Произведение эрмитовых коэффициентов ``E_x·E_y·E_z`` (нулевой порядок).

    Это **не** интеграл: у него нет нормировочного префактора. Префактор
    добавляет вызывающая функция — для перекрывания и кинетической энергии это
    ``(π/p)^{3/2}``, для кулоновских интегралов — свои (см. ниже).
    """
    return (
        _hermite_coefficient(la[0], lb[0], 0, center_a[0] - center_b[0], a, b)
        * _hermite_coefficient(la[1], lb[1], 0, center_a[1] - center_b[1], a, b)
        * _hermite_coefficient(la[2], lb[2], 0, center_a[2] - center_b[2], a, b)
    )


def _overlap_primitive(
    a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point
) -> float:
    """Перекрывание двух нормированных примитивов ``⟨a|b⟩``.

    Для s-функций на одном центре сводится к ``(π/(a+b))^{3/2}`` —
    произведению гауссовых интегралов.
    """
    p = a + b
    return float(_hermite_product(a, la, center_a, b, lb, center_b) * (math.pi / p) ** 1.5)


def _kinetic_primitive(
    a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point
) -> float:
    """Кинетический интеграл ``⟨a|−½∇²|b⟩``.

    Оператор действует на кет-функцию. Для одной оси
    ``f = (x−B)^l e^{−b(x−B)²}``::

        f″ = l(l−1)(x−B)^{l−2}e − 2b(2l+1)(x−B)^l e + 4b²(x−B)^{l+2}e

    откуда в трёх измерениях (``L = l_x+l_y+l_z`` кета)::

        T = b(2L+3)·S(l_a, l_b)
            − ½·Σ_i l_{b,i}(l_{b,i}−1)·S(l_a, l_b − 2e_i)
            − 2b²·Σ_i S(l_a, l_b + 2e_i)

    Третье слагаемое **повышает** угловой момент кета: запись с понижением
    обоих индексов даёт неверный результат (проверено квадратурой).
    """
    l_total = lb[0] + lb[1] + lb[2]

    def shifted(powers: Powers, axis: int, delta: int) -> Powers:
        updated = list(powers)
        updated[axis] += delta
        return (updated[0], updated[1], updated[2])

    total = b * (2 * l_total + 3) * _overlap_primitive(a, la, center_a, b, lb, center_b)
    for axis in range(3):
        if lb[axis] >= 2:
            total -= (
                0.5
                * lb[axis]
                * (lb[axis] - 1)
                * _overlap_primitive(a, la, center_a, b, shifted(lb, axis, -2), center_b)
            )
        total -= (
            2.0 * b * b * _overlap_primitive(a, la, center_a, b, shifted(lb, axis, 2), center_b)
        )
    return total


def _nuclear_primitive(
    a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point, nucleus: Point
) -> float:
    """Интеграл притяжения к одному ядру без множителя заряда (Helgaker 9.9.20)."""
    p = a + b
    pair_center: Point = tuple((a * center_a[axis] + b * center_b[axis]) / p for axis in range(3))  # type: ignore[assignment]
    difference: Point = tuple(pair_center[axis] - nucleus[axis] for axis in range(3))  # type: ignore[assignment]
    rpc = math.sqrt(sum(component * component for component in difference))

    total = 0.0
    for t in range(la[0] + lb[0] + 1):
        coefficient_x = _hermite_coefficient(la[0], lb[0], t, center_a[0] - center_b[0], a, b)
        if coefficient_x == 0.0:
            continue
        for u in range(la[1] + lb[1] + 1):
            coefficient_y = _hermite_coefficient(la[1], lb[1], u, center_a[1] - center_b[1], a, b)
            if coefficient_y == 0.0:
                continue
            for v in range(la[2] + lb[2] + 1):
                coefficient_z = _hermite_coefficient(
                    la[2], lb[2], v, center_a[2] - center_b[2], a, b
                )
                if coefficient_z == 0.0:
                    continue
                total += (
                    coefficient_x
                    * coefficient_y
                    * coefficient_z
                    * _hermite_coulomb(t, u, v, 0, p, difference, rpc)
                )
    return 2.0 * math.pi / p * total


def _nuclear_primitive_position_derivative(
    a: float,
    la: Powers,
    center_a: Point,
    b: float,
    lb: Powers,
    center_b: Point,
    nucleus: Point,
    axis: int,
) -> float:
    """Производная интеграла притяжения по **положению ядра** (Helgaker 9.9.21).

    Ядро зависит от положения ядра ``R_C`` только через ``P − R_C``. Поскольку
    эрмитов кулоновский интеграл по определению есть производная
    ``R^n_{tuv} = ∂^{t+u+v}R^n_{000}/∂P_x^t∂P_y^u∂P_z^v``, сдвиг индекса на
    единицу вверх и даёт производную по ``P``, а производная по ``R_C``
    отличается знаком::

        ∂/∂R_{C,axis} ⟨a|1/|r−R_C||b⟩ = −2π/p · Σ_{tuv} E_{tuv} R_{t+δ,u,v}

    Это слагаемое часто теряют: без него градиент неверен даже для H₂⁺, потому
    что движение ядра меняет не только базисные функции, но и сам оператор.
    """
    p = a + b
    pair_center: Point = tuple(  # type: ignore[assignment]
        (a * center_a[index] + b * center_b[index]) / p for index in range(3)
    )
    difference: Point = tuple(  # type: ignore[assignment]
        pair_center[index] - nucleus[index] for index in range(3)
    )
    rpc = math.sqrt(sum(component * component for component in difference))

    total = 0.0
    for t in range(la[0] + lb[0] + 1):
        coefficient_x = _hermite_coefficient(la[0], lb[0], t, center_a[0] - center_b[0], a, b)
        if coefficient_x == 0.0:
            continue
        for u in range(la[1] + lb[1] + 1):
            coefficient_y = _hermite_coefficient(la[1], lb[1], u, center_a[1] - center_b[1], a, b)
            if coefficient_y == 0.0:
                continue
            for v in range(la[2] + lb[2] + 1):
                coefficient_z = _hermite_coefficient(
                    la[2], lb[2], v, center_a[2] - center_b[2], a, b
                )
                if coefficient_z == 0.0:
                    continue
                indices = [t, u, v]
                indices[axis] += 1
                total += (
                    coefficient_x
                    * coefficient_y
                    * coefficient_z
                    * _hermite_coulomb(indices[0], indices[1], indices[2], 0, p, difference, rpc)
                )
    return -2.0 * math.pi / p * total


def _multipole_primitive(
    a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point, axis: int
) -> float:
    """Интеграл ``⟨a|r_axis|b⟩``.

    Раскладываем ``r = (r − A) + A``: первая часть даёт Эрмитов коэффициент с
    повышенным угловым моментом, вторая — ``A_axis`` на перекрывание. Потеря
    второго слагаемого — типичная ошибка, из-за которой диполь получается
    зависящим от положения начала координат.
    """
    raised = list(la)
    raised[axis] += 1
    raised_powers: Powers = (raised[0], raised[1], raised[2])

    product = 1.0
    for component in range(3):
        powers_a = raised_powers[component] if component == axis else la[component]
        product *= _hermite_coefficient(
            powers_a, lb[component], 0, center_a[component] - center_b[component], a, b
        )
    # Оба слагаемых — произведения эрмитовых коэффициентов, поэтому общий
    # префактор ``(π/p)^{3/2}`` выносится за скобку. Применять его только к
    # одному из слагаемых (например, через уже нормированное перекрывание)
    # нельзя: получится смесь величин разного масштаба.
    prefactor = (math.pi / (a + b)) ** 1.5
    return float(
        (product + center_a[axis] * _hermite_product(a, la, center_a, b, lb, center_b)) * prefactor
    )


def _eri_primitive(
    a: float,
    la: Powers,
    center_a: Point,
    b: float,
    lb: Powers,
    center_b: Point,
    c: float,
    lc: Powers,
    center_c: Point,
    d: float,
    ld: Powers,
    center_d: Point,
) -> float:
    """Двухэлектронный интеграл ``(ab|cd)`` для четырёх нормированных примитивов."""
    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    pair_p: Point = tuple((a * center_a[axis] + b * center_b[axis]) / p for axis in range(3))  # type: ignore[assignment]
    pair_q: Point = tuple((c * center_c[axis] + d * center_d[axis]) / q for axis in range(3))  # type: ignore[assignment]
    difference: Point = tuple(pair_p[axis] - pair_q[axis] for axis in range(3))  # type: ignore[assignment]
    rpq = math.sqrt(sum(component * component for component in difference))

    def hermite_set(
        l1: Powers,
        l2: Powers,
        center1: Point,
        center2: Point,
        ex1: float,
        ex2: float,
        alternating: bool,
    ) -> dict[Powers, float]:
        result: dict[Powers, float] = {}
        for t in range(l1[0] + l2[0] + 1):
            for u in range(l1[1] + l2[1] + 1):
                for v in range(l1[2] + l2[2] + 1):
                    value = (
                        _hermite_coefficient(l1[0], l2[0], t, center1[0] - center2[0], ex1, ex2)
                        * _hermite_coefficient(l1[1], l2[1], u, center1[1] - center2[1], ex1, ex2)
                        * _hermite_coefficient(l1[2], l2[2], v, center1[2] - center2[2], ex1, ex2)
                    )
                    if value == 0.0:
                        continue
                    if alternating and (t + u + v) % 2:
                        value = -value
                    result[(t, u, v)] = value
        return result

    hermite_ab = hermite_set(la, lb, center_a, center_b, a, b, alternating=False)
    if not hermite_ab:
        return 0.0
    hermite_cd = hermite_set(lc, ld, center_c, center_d, c, d, alternating=True)
    if not hermite_cd:
        return 0.0

    total = 0.0
    for (t1, u1, v1), coefficient_ab in hermite_ab.items():
        for (t2, u2, v2), coefficient_cd in hermite_cd.items():
            total += (
                coefficient_ab
                * coefficient_cd
                * _hermite_coulomb(t1 + t2, u1 + u2, v1 + v2, 0, alpha, difference, rpq)
            )
    return float(2.0 * PI_5_2 / (p * q * math.sqrt(p + q)) * total)


# --------------------------------------------------------------------------- #
# Сборка по базису
# --------------------------------------------------------------------------- #
def _shell_centers(basis: BasisSet, molecule: Molecule) -> list[Point]:
    """Координаты центров оболочек в борах."""
    geometry: list[Point] = [
        tuple(angstrom_to_bohr(value) for value in atom.position)  # type: ignore[misc]
        for atom in molecule.atoms
    ]
    return [geometry[shell.center] for shell in basis.shells]


def _shell_offset(basis: BasisSet, index: int) -> int:
    return sum(shell.n_cartesian for shell in basis.shells[:index])


def _place_block(matrix: np.ndarray, basis: BasisSet, i: int, j: int, block: np.ndarray) -> None:
    """Записывает блок оболочки в симметричную матрицу (обе треугольные части)."""
    offset_i = _shell_offset(basis, i)
    offset_j = _shell_offset(basis, j)
    rows = slice(offset_i, offset_i + block.shape[0])
    columns = slice(offset_j, offset_j + block.shape[1])
    matrix[rows, columns] = block
    if i != j:
        matrix[columns, rows] = block.T


def _pair_block(
    shell_a: Shell, center_a: Point, shell_b: Shell, center_b: Point, kernel: PrimitiveKernel
) -> np.ndarray:
    powers_a = cartesian_powers(shell_a.angular_momentum)
    powers_b = cartesian_powers(shell_b.angular_momentum)
    scales_a = shell_a.component_scales
    scales_b = shell_b.component_scales
    block = np.zeros((len(powers_a), len(powers_b)))
    for row, pa in enumerate(powers_a):
        for column, pb in enumerate(powers_b):
            total = 0.0
            for alpha, coefficient_a in zip(shell_a.exponents, shell_a.coefficients, strict=True):
                for beta, coefficient_b in zip(
                    shell_b.exponents, shell_b.coefficients, strict=True
                ):
                    total += (
                        coefficient_a
                        * coefficient_b
                        * kernel(alpha, pa, center_a, beta, pb, center_b)
                    )
            block[row, column] = total * scales_a[row] * scales_b[column]
    return block


def build_overlap(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """Матрица перекрывания ``S``."""
    centers = _shell_centers(basis, molecule)
    matrix = np.zeros((basis.n_functions, basis.n_functions))
    for i, shell_a in enumerate(basis.shells):
        for j, shell_b in enumerate(basis.shells[: i + 1]):
            block = _pair_block(shell_a, centers[i], shell_b, centers[j], _overlap_primitive)
            _place_block(matrix, basis, i, j, block)
    return matrix


def build_kinetic(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """Матрица кинетической энергии ``T``."""
    centers = _shell_centers(basis, molecule)
    matrix = np.zeros((basis.n_functions, basis.n_functions))
    for i, shell_a in enumerate(basis.shells):
        for j, shell_b in enumerate(basis.shells[: i + 1]):
            block = _pair_block(shell_a, centers[i], shell_b, centers[j], _kinetic_primitive)
            _place_block(matrix, basis, i, j, block)
    return matrix


def build_nuclear_attraction(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """Матрица притяжения электронов к ядрам ``V`` (отрицательная)."""
    centers = _shell_centers(basis, molecule)
    unique_nuclei: list[tuple[int, Point]] = [
        (
            atom.z,
            tuple(angstrom_to_bohr(value) for value in atom.position),  # type: ignore[misc]
        )
        for atom in molecule.atoms
    ]

    def kernel(
        a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point
    ) -> float:
        total = 0.0
        for charge, position in unique_nuclei:
            total -= charge * _nuclear_primitive(a, la, center_a, b, lb, center_b, position)
        return total

    matrix = np.zeros((basis.n_functions, basis.n_functions))
    for i, shell_a in enumerate(basis.shells):
        for j, shell_b in enumerate(basis.shells[: i + 1]):
            block = _pair_block(shell_a, centers[i], shell_b, centers[j], kernel)
            _place_block(matrix, basis, i, j, block)
    return matrix


def build_core_hamiltonian(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """Одноэлектронный гамильтониан ``H = T + V``."""
    return np.asarray(build_kinetic(basis, molecule) + build_nuclear_attraction(basis, molecule))


def build_electron_repulsion(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """Тензор двухэлектронных интегралов ``(μν|λσ)`` формы ``(n, n, n, n)``.

    Вычисляются только уникальные квартеты оболочек (условие
    ``(ij) ≥ (kl)`` в лексикографическом порядке пар), остальные восемь
    перестановок заполняются симметрией. Константа ~1/8 при той же асимптотике.

    Квартеты одного класса (:class:`_ClassGeometry`) считаются пачками: это не
    приближение, а другой порядок обхода — значения совпадают с поквартиетной
    сборкой :func:`_quartet_block` с точностью до порядка суммирования.
    """
    shells = basis.shells
    centers = _shell_centers(basis, molecule)
    offsets = _shell_offsets(basis)
    tensor = np.zeros((basis.n_functions,) * 4)

    for quartets in _group_by_class(shells, _unique_quartets(len(shells))).values():
        geometry = _class_geometry(shells, quartets[0])
        budget = geometry.angular_budget
        step = _batch_step(geometry, budget)
        for start in range(0, len(quartets), step):
            chunk = quartets[start : start + step]
            arrays = _batch_arrays(shells, centers, chunk, angular_budget=budget)
            blocks = _batch_eri_blocks(arrays, geometry)
            _place_quartet_batch(tensor, offsets, chunk, geometry, blocks)
    return tensor


def _quartet_block_scalar(
    shell_a: Shell,
    center_a: Point,
    shell_b: Shell,
    center_b: Point,
    shell_c: Shell,
    center_c: Point,
    shell_d: Shell,
    center_d: Point,
) -> np.ndarray:
    """Блок ``(ab|cd)``, считанный «в лоб» по примитивным квартетам.

    Рабочий путь — векторизованный :func:`_quartet_block`; эта функция
    оставлена как читаемая спецификация формул и как оракул для теста на
    совпадение двух путей.
    """
    powers_a = cartesian_powers(shell_a.angular_momentum)
    powers_b = cartesian_powers(shell_b.angular_momentum)
    powers_c = cartesian_powers(shell_c.angular_momentum)
    powers_d = cartesian_powers(shell_d.angular_momentum)
    scales = (
        shell_a.component_scales,
        shell_b.component_scales,
        shell_c.component_scales,
        shell_d.component_scales,
    )
    block = np.zeros((len(powers_a), len(powers_b), len(powers_c), len(powers_d)))
    for ia, pa in enumerate(powers_a):
        for ib, pb in enumerate(powers_b):
            for ic, pc in enumerate(powers_c):
                for idx, pd in enumerate(powers_d):
                    total = 0.0
                    for alpha, ca in zip(shell_a.exponents, shell_a.coefficients, strict=True):
                        for beta, cb in zip(shell_b.exponents, shell_b.coefficients, strict=True):
                            for gamma, cc in zip(
                                shell_c.exponents, shell_c.coefficients, strict=True
                            ):
                                for delta, cd in zip(
                                    shell_d.exponents, shell_d.coefficients, strict=True
                                ):
                                    total += (
                                        ca
                                        * cb
                                        * cc
                                        * cd
                                        * _eri_primitive(
                                            alpha,
                                            pa,
                                            center_a,
                                            beta,
                                            pb,
                                            center_b,
                                            gamma,
                                            pc,
                                            center_c,
                                            delta,
                                            pd,
                                            center_d,
                                        )
                                    )
                    block[ia, ib, ic, idx] = total * (
                        scales[0][ia] * scales[1][ib] * scales[2][ic] * scales[3][idx]
                    )
    return block


def _place_quartet(
    tensor: np.ndarray, basis: BasisSet, i: int, j: int, k: int, m: int, block: np.ndarray
) -> None:
    oi, oj, ok, om = (_shell_offset(basis, index) for index in (i, j, k, m))
    si = slice(oi, oi + basis.shells[i].n_cartesian)
    sj = slice(oj, oj + basis.shells[j].n_cartesian)
    sk = slice(ok, ok + basis.shells[k].n_cartesian)
    sm = slice(om, om + basis.shells[m].n_cartesian)

    tensor[si, sj, sk, sm] = block
    tensor[sj, si, sk, sm] = block.transpose(1, 0, 2, 3)
    tensor[si, sj, sm, sk] = block.transpose(0, 1, 3, 2)
    tensor[sj, si, sm, sk] = block.transpose(1, 0, 3, 2)
    tensor[sk, sm, si, sj] = block.transpose(2, 3, 0, 1)
    tensor[sm, sk, si, sj] = block.transpose(3, 2, 0, 1)
    tensor[sk, sm, sj, si] = block.transpose(2, 3, 1, 0)
    tensor[sm, sk, sj, si] = block.transpose(3, 2, 1, 0)


def _place_pair_full(
    matrix: np.ndarray, basis: BasisSet, i: int, j: int, block: np.ndarray
) -> None:
    """Записывает блок **без** зеркального отражения.

    Производная интеграла по центру бра-функции несимметрична:
    ``∂S_μν/∂A_μ ≠ ∂S_νμ/∂A_ν``. Применять здесь обычное
    :func:`_place_block` — значит подставить в нижний треугольник чужие
    значения; именно так градиент теряет поступательную инвариантность.
    """
    offset_i = _shell_offset(basis, i)
    offset_j = _shell_offset(basis, j)
    matrix[offset_i : offset_i + block.shape[0], offset_j : offset_j + block.shape[1]] = block


def bra_derivative_kernel(kernel: PrimitiveKernel, axis: int) -> PrimitiveKernel:
    r"""Ядро производной интеграла по центру бра-функции.

    Для примитива ``g = (x−A)^l e^{−α(x−A)²}``::

        ∂g/∂A_x = 2α·g(l+1) − l·g(l−1)

    поэтому производная интеграла выражается **тем же** интегралом со сдвинутыми
    угловыми моментами — новой математики не требуется (проверено численно:
    согласие с конечными разностями до 1e-9 на всех четырёх центрах ERI).
    """

    def derived(
        a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point
    ) -> float:
        raised = list(la)
        raised[axis] += 1
        total = 2.0 * a * kernel(a, (raised[0], raised[1], raised[2]), center_a, b, lb, center_b)
        if la[axis] > 0:
            lowered = list(la)
            lowered[axis] -= 1
            total -= la[axis] * kernel(
                a, (lowered[0], lowered[1], lowered[2]), center_a, b, lb, center_b
            )
        return total

    return derived


def build_overlap_derivative(basis: BasisSet, molecule: Molecule, axis: int) -> np.ndarray:
    r"""``∂S_μν/∂A_x``, где ``A`` — центр **бра**-оболочки функции ``μ``.

    Производная по центру кета не вычисляется отдельно: ``S`` симметрична,
    поэтому она равна транспонированной матрице. То же верно для ``T`` и
    электронной части ``V``.
    """
    centers = _shell_centers(basis, molecule)
    kernel = bra_derivative_kernel(_overlap_primitive, axis)
    matrix = np.zeros((basis.n_functions, basis.n_functions))
    # Полный перебор пар: производная несимметрична, зеркалить нельзя.
    for i, shell_a in enumerate(basis.shells):
        for j, shell_b in enumerate(basis.shells):
            block = _pair_block(shell_a, centers[i], shell_b, centers[j], kernel)
            _place_pair_full(matrix, basis, i, j, block)
    return matrix


def build_kinetic_derivative(basis: BasisSet, molecule: Molecule, axis: int) -> np.ndarray:
    """``∂T_μν/∂A_x`` по центру бра-оболочки."""
    centers = _shell_centers(basis, molecule)
    kernel = bra_derivative_kernel(_kinetic_primitive, axis)
    matrix = np.zeros((basis.n_functions, basis.n_functions))
    # Полный перебор пар: производная несимметрична, зеркалить нельзя.
    for i, shell_a in enumerate(basis.shells):
        for j, shell_b in enumerate(basis.shells):
            block = _pair_block(shell_a, centers[i], shell_b, centers[j], kernel)
            _place_pair_full(matrix, basis, i, j, block)
    return matrix


def build_nuclear_attraction_center_derivative(
    basis: BasisSet, molecule: Molecule, axis: int
) -> np.ndarray:
    """``∂V_μν/∂A_x`` по центру бра-оболочки (без вклада движения самих ядер)."""
    centers = _shell_centers(basis, molecule)
    unique_nuclei: list[tuple[int, Point]] = [
        (
            atom.z,
            tuple(angstrom_to_bohr(value) for value in atom.position),  # type: ignore[misc]
        )
        for atom in molecule.atoms
    ]

    def kernel(
        a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point
    ) -> float:
        total = 0.0
        for charge, position in unique_nuclei:
            total -= charge * _nuclear_primitive(a, la, center_a, b, lb, center_b, position)
        return total

    derived = bra_derivative_kernel(kernel, axis)
    matrix = np.zeros((basis.n_functions, basis.n_functions))
    for i, shell_a in enumerate(basis.shells):
        for j, shell_b in enumerate(basis.shells):
            block = _pair_block(shell_a, centers[i], shell_b, centers[j], derived)
            _place_pair_full(matrix, basis, i, j, block)
    return matrix


def build_nuclear_attraction_position_derivative(
    basis: BasisSet, molecule: Molecule, atom_index: int, axis: int
) -> np.ndarray:
    r"""``∂V_μν/∂R_{A,x}`` — слагаемое от движения самого ядра ``A``.

    Симметричная матрица (зависит только от оператора, а не от того, какая из
    двух функций дифференцируется), поэтому заполняются обе треугольные части.
    """
    centers = _shell_centers(basis, molecule)
    atom = molecule.atoms[atom_index]
    origin = atom.position
    position: Point = (
        angstrom_to_bohr(origin[0]),
        angstrom_to_bohr(origin[1]),
        angstrom_to_bohr(origin[2]),
    )
    charge = atom.z

    def kernel(
        a: float, la: Powers, center_a: Point, b: float, lb: Powers, center_b: Point
    ) -> float:
        return -charge * _nuclear_primitive_position_derivative(
            a, la, center_a, b, lb, center_b, position, axis
        )

    matrix = np.zeros((basis.n_functions, basis.n_functions))
    for i, shell_a in enumerate(basis.shells):
        for j, shell_b in enumerate(basis.shells[: i + 1]):
            block = _pair_block(shell_a, centers[i], shell_b, centers[j], kernel)
            _place_block(matrix, basis, i, j, block)
    return matrix


def _eri_derivative_primitive(
    axis: int,
    a: float,
    la: Powers,
    center_a: Point,
    b: float,
    lb: Powers,
    center_b: Point,
    c: float,
    lc: Powers,
    center_c: Point,
    d: float,
    ld: Powers,
    center_d: Point,
) -> float:
    """``∂(ab|cd)/∂A_x`` — производная по центру первой бра-функции."""
    raised = list(la)
    raised[axis] += 1
    total = (
        2.0
        * a
        * _eri_primitive(
            a,
            (raised[0], raised[1], raised[2]),
            center_a,
            b,
            lb,
            center_b,
            c,
            lc,
            center_c,
            d,
            ld,
            center_d,
        )
    )
    if la[axis] > 0:
        lowered = list(la)
        lowered[axis] -= 1
        total -= la[axis] * _eri_primitive(
            a,
            (lowered[0], lowered[1], lowered[2]),
            center_a,
            b,
            lb,
            center_b,
            c,
            lc,
            center_c,
            d,
            ld,
            center_d,
        )
    return total


def build_electron_repulsion_derivative(
    basis: BasisSet, molecule: Molecule, axis: int
) -> np.ndarray:
    r"""``∂(μν|λσ)/∂A_x``, где ``A`` — центр оболочки функции ``μ``.

    Производные по трём остальным центрам получаются из этой одной сборкой
    перестановкой индексов — но **не** по 8-кратной симметрии самого тензора:
    у производной по центру μ она не выполняется. Используются только
    тождества ``(μν|λσ) = (νμ|λσ) = (λσ|μν)``, применённые к производной::

        по центру ν: transpose(1, 0, 2, 3)
        по центру λ: transpose(2, 3, 0, 1)
        по центру σ: transpose(2, 3, 1, 0)

    Поэтому стоимость градиента — три сборки тензора (по одной на ось), а не
    двенадцать. Соотношения проверяются тестом, а не принимаются на веру.
    """
    centers = _shell_centers(basis, molecule)
    shells = basis.shells
    offsets = _shell_offsets(basis)
    n_shells = len(basis.shells)
    tensor = np.zeros((basis.n_functions,) * 4)

    # Из восьми перестановок тензора ERI у производной по центру μ сохраняется
    # только симметрия λ↔σ (из (μν|λσ) = (μν|σλ)). Остальные применять нельзя:
    # ∂(νμ|λσ)/∂A_ν — это производная по другому центру.
    for quartets in _group_by_class(shells, _bra_quartets(n_shells)).values():
        geometry = _class_geometry(shells, quartets[0])
        budget = geometry.angular_budget + 1
        step = _batch_step(geometry, budget)
        for start in range(0, len(quartets), step):
            chunk = quartets[start : start + step]
            arrays = _batch_arrays(shells, centers, chunk, angular_budget=budget)
            blocks = _batch_eri_derivative_blocks(axis, arrays, geometry)
            _place_quartet_derivative_batch(tensor, offsets, chunk, geometry, blocks)
    return tensor


# --------------------------------------------------------------------------- #
# Векторизованная сборка производных ERI
# --------------------------------------------------------------------------- #
# Скалярный путь (_quartet_derivative_block_scalar) остаётся в коде как читаемая
# спецификация формул и как оракул для теста: он медленный, но его правильность
# видна из записи. Рабочий путь ниже считает то же самое, но сразу по всем
# примитивным квартетам оболочки — иначе стоимость градиента определяется
# накладными расходами интерпретатора, а не арифметикой.


def _erf_array(values: np.ndarray) -> np.ndarray:
    """``erf`` для массива.

    В NumPy функции ошибок нет, а рациональные аппроксимации дают ~1e-7 —
    для квантовой химии этого мало. Поэтому используется точный ``math.erf``;
    вызовов немного: только элементы с большим аргументом.
    """
    # ``tolist`` отдаёт обычные ``float``: итерация по элементам ndarray с
    # последующим ``float()`` вчетверо медленнее на больших массивах.
    flat = values.ravel().tolist()
    return np.array([math.erf(value) for value in flat]).reshape(values.shape)


def _series_terms(x_max: float) -> int:
    r"""Число членов ряда Бойса, достаточное при ``x ≤ x_max``.

    Ряд ``F_n(x) = Σ_k (−x)^k/(k!·(2n+2k+1))`` знакочередующийся, и начиная с
    ``k > x`` его члены убывают, поэтому остаток не превосходит первого
    отброшенного члена. Берём худший порядок ``n = 0`` и возвращаем первое
    ``K``, на котором ``x^K/(K!·(2K+1)) < 1e-17``: ряд суммируется до
    ``K − 1`` включительно, то есть ``K`` — ровно первый отброшенный член.
    Запаса в один член здесь намеренно нет: минимальность проверяется тестом,
    иначе «с запасом» и «необходимо» стали бы неразличимы.

    Прежнее фиксированное число членов (25) было недостаточным: при ``x = 4``
    ошибка достигала 1.2e-12, а при ``x → 6`` — 2.8e-8, то есть на границе
    ветвей функция Бойса теряла семь знаков. Проверено сравнением с
    независимым эталоном (квадратура Гаусса–Лежандра определяющего интеграла).
    """
    terms = 1
    while terms < 200:
        magnitude = x_max**terms / (math.factorial(terms) * (2 * terms + 1))
        if magnitude < _BOYS_SERIES_TOLERANCE:
            return terms
        terms += 1
    raise ValueError(f"ряд Бойса не сошёлся за 200 членов при x = {x_max}")


def _boys_table(limit: int, x: np.ndarray) -> list[np.ndarray]:
    r"""``F_0(x) … F_limit(x)`` одним проходом по массиву.

    Тот же алгоритм, что и в скалярном :func:`_boys`: ряд Тейлора при ``x < 6``
    и рекурсия вверх от ``F_0 = ½√(π/x)·erf(√x)`` при больших ``x``.

    Все порядки считаются вместе по двум причинам. Во-первых, слагаемые ряда
    ``(−x)^k/k!`` не зависят от ``n``: прежний код пересобирал их на каждый
    порядок, то есть ``limit+1`` раз вместо одного. Во-вторых, на ветви больших
    ``x`` дорогой скалярный ``math.erf`` теперь вызывается один раз на точку, а
    старшие порядки получаются рекурсией ``F_{n+1} = ((2n+1)F_n − e^{−x})/2x``.
    Для квартета ``d``-оболочек ``limit = 8``, поэтому экономия девятикратная.
    """
    result = [np.empty_like(x) for _ in range(limit + 1)]
    small = x < _BOYS_SERIES_LIMIT
    if np.any(small):
        xs = x[small]
        term = np.ones_like(xs)
        # Член k = 0 равен 1/(2n+1), а не единице: при n > 0 это существенно.
        series = [term / (2.0 * n + 1.0) for n in range(limit + 1)]
        for k in range(1, _series_terms(float(xs.max()))):
            term = term * (-xs / k)
            for n in range(limit + 1):
                series[n] = series[n] + term / (2 * n + 2 * k + 1)
        for n in range(limit + 1):
            result[n][small] = series[n]
    large = ~small
    if np.any(large):
        xl = x[large]
        value = 0.5 * np.sqrt(math.pi / xl) * _erf_array(np.sqrt(xl))
        exponent = np.exp(-xl)
        result[0][large] = value
        for n in range(1, limit + 1):
            value = ((2 * n - 1) * value - exponent) / (2.0 * xl)
            result[n][large] = value
    return result


def _boys_array(order: int, x: np.ndarray) -> np.ndarray:
    r"""``F_n(x)`` сразу для массива аргументов.

    Тонкая обёртка над :func:`_boys_table`: совпадение со скалярным
    :func:`_boys` на обеих ветвях проверяется тестом.
    """
    return _boys_table(order, x)[order]


def _hermite_1d(
    i: int, j: int, q: float | np.ndarray, a: np.ndarray, b: np.ndarray
) -> list[np.ndarray]:
    r"""``E^{ij}_t`` для одной декартовой оси как список массивов по квартетам.

    Та же рекурсия, что в :func:`_hermite_coefficient`, но ``a`` и ``b`` —
    массивы, поэтому за один проход получаются коэффициенты всех примитивных
    квартетов оболочки::

        E^{i+1,j}_t = E^{i,j}_{t−1}/(2p) + (t+1)·E^{i,j}_{t+1} − (bQ/p)·E^{i,j}_t
        E^{i,j+1}_t = E^{i,j}_{t−1}/(2p) + (t+1)·E^{i,j}_{t+1} + (aQ/p)·E^{i,j}_t
        E^{00}_0    = exp(−μQ²),  μ = ab/p

    ``q`` — либо скаляр (один квартет оболочек), либо массив, расширяемый до
    формы ``a``: в пакетном пути каждой пачке квартетов соответствует свой
    вектор ``A − B``, поэтому здесь он имеет вид ``(N, 1, 1, …)``.
    """
    p = a + b
    table: dict[tuple[int, int], list[np.ndarray]] = {(0, 0): [np.exp(-(a * b / p) * q * q)]}
    for total in range(1, i + j + 1):
        for first in range(total + 1):
            second = total - first
            if first > 0:
                previous = table[(first - 1, second)]
                shift = b * q / p
                sign = -1.0
            else:
                previous = table[(first, second - 1)]
                shift = a * q / p
                sign = 1.0
            highest = len(previous) - 1
            values: list[np.ndarray] = []
            for t in range(total + 1):
                # E^{ij}_t = 0 при t > i + j, то есть вне границ предыдущего
                # уровня рекурсии; верхний индекс t = total как раз такой.
                lower = previous[t - 1] / (2.0 * p) if t >= 1 else 0.0
                same = previous[t] if t <= highest else 0.0
                upper = (t + 1) * previous[t + 1] if t + 1 <= highest else 0.0
                values.append(lower + sign * shift * same + upper)
            table[(first, second)] = values
    return table[(i, j)]


def _coulomb_table(
    limit: int,
    alpha: np.ndarray,
    pqx: np.ndarray,
    pqy: np.ndarray,
    pqz: np.ndarray,
    x: np.ndarray,
) -> dict[tuple[int, int, int, int], np.ndarray]:
    r"""``R^n_{tuv}`` для всех ``t+u+v+n ≤ limit`` как массивы по квартетам.

    Рекурсия Хельгакера (9.9.15–9.9.18)::

        R^n_{000}  = (−2α)^n F_n(α R²)
        R^n_{tuv}  = PQ_x R^{n+1}_{t−1,u,v} + (t−1) R^{n+1}_{t−2,u,v}   (t > 0)

    и циклически по ``u``, ``v``. Индекс ``n`` растёт на единицу на каждом шаге
    рекурсии, поэтому при ``t+u+v = k`` довольно ``n ≤ limit − k``: обход по
    возрастающему ``k`` гарантирует, что нужные значения уже готовы.
    """
    table: dict[tuple[int, int, int, int], np.ndarray] = {}
    boys = _boys_table(limit, x)
    factor = np.ones_like(alpha)
    for n in range(limit + 1):
        table[(0, 0, 0, n)] = factor * boys[n]
        factor = factor * (-2.0 * alpha)
    for k in range(1, limit + 1):
        for t in range(k + 1):
            for u in range(k - t + 1):
                v = k - t - u
                for n in range(limit - k + 1):
                    if t > 0:
                        value = pqx * table[(t - 1, u, v, n + 1)]
                        if t > 1:
                            value = value + (t - 1) * table[(t - 2, u, v, n + 1)]
                    elif u > 0:
                        value = pqy * table[(t, u - 1, v, n + 1)]
                        if u > 1:
                            value = value + (u - 1) * table[(t, u - 2, v, n + 1)]
                    else:
                        value = pqz * table[(t, u, v - 1, n + 1)]
                        if v > 1:
                            value = value + (v - 1) * table[(t, u, v - 2, n + 1)]
                    table[(t, u, v, n)] = value
    return table


def _hermite_product_arrays(
    powers_a: Powers,
    powers_b: Powers,
    qx: float | np.ndarray,
    qy: float | np.ndarray,
    qz: float | np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    alternating: bool = False,
    cache: dict[tuple[int, int, int], list[np.ndarray]] | None = None,
) -> dict[Powers, np.ndarray]:
    """``E^{ab}_{tuv} = E_x·E_y·E_z`` как словарь по ``(t, u, v)``.

    При ``alternating=True`` коэффициент умножается на ``(−1)^{t+u+v}`` — это
    кет-сторона кулоновского интеграла.

    ``cache`` — общий для стороны словарь одномерных рекурсий с ключом
    ``(ось, i, j)``. У декартовых компонент одной оболочки степени повторяются:
    у ``p``-оболочки на одну ось приходятся пары ``(1,0)``, ``(0,0)``, ``(0,0)``,
    и без кеша рекурсия считалась бы заново на каждую пару компонент. Кеш
    ничего не приближает — возвращаются ровно те же массивы.
    """
    if cache is None:
        cache = {}
    axes: list[list[np.ndarray]] = []
    for axis, (first, second, q) in enumerate(
        (
            (powers_a[0], powers_b[0], qx),
            (powers_a[1], powers_b[1], qy),
            (powers_a[2], powers_b[2], qz),
        )
    ):
        key = (axis, first, second)
        if key not in cache:
            cache[key] = _hermite_1d(first, second, q, a, b)
        axes.append(cache[key])
    result: dict[Powers, np.ndarray] = {}
    for t, ex in enumerate(axes[0]):
        for u, ey in enumerate(axes[1]):
            for v, ez in enumerate(axes[2]):
                product = ex * ey * ez
                if alternating and (t + u + v) % 2:
                    product = -product
                if not np.any(product):
                    continue
                result[(t, u, v)] = product
    return result


def _contract(
    bra: dict[Powers, np.ndarray],
    ket: dict[Powers, np.ndarray],
    coulomb: dict[tuple[int, int, int, int], np.ndarray],
) -> np.ndarray:
    """``Σ E^{ab}_{tuv}·E^{cd}_{t'u'v'}·R^0_{t+t',u+u',v+v'}`` по квартетам."""
    total = np.zeros_like(coulomb[(0, 0, 0, 0)])
    for (t1, u1, v1), coefficient_bra in bra.items():
        for (t2, u2, v2), coefficient_ket in ket.items():
            total = (
                total + coefficient_bra * coefficient_ket * coulomb[(t1 + t2, u1 + u2, v1 + v2, 0)]
            )
    return total


@dataclass(frozen=True, slots=True)
class _QuartetArrays:
    """Всё, что зависит от примитивного квартета, но не от угловых моментов.

    Собирается один раз на квартет оболочек и переиспользуется для всех
    декартовых компонент. Именно повторный расчёт этих величин на каждую
    компоненту и определял стоимость интегралов: на воде/6-31G было
    147 тыс. полных проходов ``_eri_primitive`` при 13 базисных функциях.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    coefficients: np.ndarray
    prefactor: np.ndarray
    coulomb: dict[tuple[int, int, int, int], np.ndarray]
    q: tuple[float, float, float]
    r: tuple[float, float, float]


def _quartet_arrays(
    shell_a: Shell,
    center_a: Point,
    shell_b: Shell,
    center_b: Point,
    shell_c: Shell,
    center_c: Point,
    shell_d: Shell,
    center_d: Point,
    *,
    angular_budget: int,
) -> _QuartetArrays:
    """Раскладывает квартет оболочек по всем примитивным квартетам сразу.

    Порядок примитивов — прямое произведение ``(α, β, γ, δ)`` с медленнее всего
    меняющимся ``α``; массив коэффициентов сжатия строится в том же порядке,
    поэтому свёртка ``coefficients @ (…)`` корректна.

    ``angular_budget`` — максимальная сумма ``t+u+v`` в таблице ``R``: для
    энергии это ``l_a+l_b+l_c+l_d``, для производной на единицу больше.
    """
    counts = [
        len(shell_a.exponents),
        len(shell_b.exponents),
        len(shell_c.exponents),
        len(shell_d.exponents),
    ]
    a = np.repeat(np.asarray(shell_a.exponents, dtype=float), counts[1] * counts[2] * counts[3])
    b = np.repeat(
        np.tile(np.asarray(shell_b.exponents, dtype=float), counts[0]), counts[2] * counts[3]
    )
    c = np.repeat(
        np.tile(np.asarray(shell_c.exponents, dtype=float), counts[0] * counts[1]), counts[3]
    )
    d = np.tile(np.asarray(shell_d.exponents, dtype=float), counts[0] * counts[1] * counts[2])
    coefficients = np.einsum(
        "i,j,k,l->ijkl",
        np.asarray(shell_a.coefficients, dtype=float),
        np.asarray(shell_b.coefficients, dtype=float),
        np.asarray(shell_c.coefficients, dtype=float),
        np.asarray(shell_d.coefficients, dtype=float),
    ).reshape(-1)

    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    pqx = (a * center_a[0] + b * center_b[0]) / p - (c * center_c[0] + d * center_d[0]) / q
    pqy = (a * center_a[1] + b * center_b[1]) / p - (c * center_c[1] + d * center_d[1]) / q
    pqz = (a * center_a[2] + b * center_b[2]) / p - (c * center_c[2] + d * center_d[2]) / q
    x = alpha * (pqx * pqx + pqy * pqy + pqz * pqz)

    return _QuartetArrays(
        a=a,
        b=b,
        c=c,
        d=d,
        coefficients=coefficients,
        prefactor=2.0 * PI_5_2 / (p * q * np.sqrt(p + q)),
        coulomb=_coulomb_table(angular_budget, alpha, pqx, pqy, pqz, x),
        q=(center_a[0] - center_b[0], center_a[1] - center_b[1], center_a[2] - center_b[2]),
        r=(center_c[0] - center_d[0], center_c[1] - center_d[1], center_c[2] - center_d[2]),
    )


def _quartet_block(
    shell_a: Shell,
    center_a: Point,
    shell_b: Shell,
    center_b: Point,
    shell_c: Shell,
    center_c: Point,
    shell_d: Shell,
    center_d: Point,
) -> np.ndarray:
    """Блок ``(ab|cd)`` сразу по всем примитивным квартетам оболочки.

    Формула та же, что в скалярном :func:`_quartet_block_scalar`; отличается
    только способ счёта. Кет-сторона и таблица ``R^n_{tuv}`` не зависят от
    угловых моментов бра-функции, поэтому считаются один раз.
    """
    powers_a = cartesian_powers(shell_a.angular_momentum)
    powers_b = cartesian_powers(shell_b.angular_momentum)
    powers_c = cartesian_powers(shell_c.angular_momentum)
    powers_d = cartesian_powers(shell_d.angular_momentum)
    scales = (
        shell_a.component_scales,
        shell_b.component_scales,
        shell_c.component_scales,
        shell_d.component_scales,
    )
    arrays = _quartet_arrays(
        shell_a,
        center_a,
        shell_b,
        center_b,
        shell_c,
        center_c,
        shell_d,
        center_d,
        angular_budget=(
            shell_a.angular_momentum
            + shell_b.angular_momentum
            + shell_c.angular_momentum
            + shell_d.angular_momentum
        ),
    )
    qx, qy, qz = arrays.q
    rx, ry, rz = arrays.r

    # Бра-коэффициенты зависят только от пары (A, B), кет-коэффициенты — только
    # от пары (C, D). Прежний цикл вычислял бра внутри обхода по ic, idx и ib,
    # то есть |A|·|B|·|C|·|D| раз вместо |A|·|B|: для квартета d-оболочек это
    # 1296 вычислений вместо 36, и каждый вызов — это три рекурсии Хельгакера
    # по малым массивам, где стоимость определяет накладной расход NumPy, а не
    # арифметика. Кэширование повторяет одни и те же вызовы, поэтому числа
    # совпадают бит в бит; меняется только их количество.
    ket_table = {
        (ic, idx): _hermite_product_arrays(pc, pd, rx, ry, rz, arrays.c, arrays.d, alternating=True)
        for ic, pc in enumerate(powers_c)
        for idx, pd in enumerate(powers_d)
    }
    # Если кет-сторона пуста целиком, бра считать незачем: прежний цикл в этом
    # случае не доходил до неё ни разу.
    if not any(ket_table.values()):
        return np.zeros((len(powers_a), len(powers_b), len(powers_c), len(powers_d)))
    bra_table = {
        (ia, ib): _hermite_product_arrays(pa, pb, qx, qy, qz, arrays.a, arrays.b)
        for ia, pa in enumerate(powers_a)
        for ib, pb in enumerate(powers_b)
    }

    block = np.zeros((len(powers_a), len(powers_b), len(powers_c), len(powers_d)))
    for (ic, idx), ket in ket_table.items():
        if not ket:
            continue
        for (ia, ib), bra in bra_table.items():
            values = _contract(bra, ket, arrays.coulomb)
            block[ia, ib, ic, idx] = float(arrays.coefficients @ (arrays.prefactor * values)) * (
                scales[0][ia] * scales[1][ib] * scales[2][ic] * scales[3][idx]
            )
    return block


def _quartet_derivative_block(
    axis: int,
    shell_a: Shell,
    center_a: Point,
    shell_b: Shell,
    center_b: Point,
    shell_c: Shell,
    center_c: Point,
    shell_d: Shell,
    center_d: Point,
) -> np.ndarray:
    """Блок ``∂(ab|cd)/∂A_x`` сразу по всем примитивным квартетам оболочки.

    Формула та же, что в скалярном пути::

        ∂(ab|cd)/∂A_x = 2a·(a+e_x, b|c,d) − l_a,x·(a−e_x, b|c,d)

    Отличается только способ счёта: геометрия квартета, эрмитовы коэффициенты
    кет-стороны и таблица ``R^n_{tuv}`` вычисляются один раз на квартет, а не
    заново для каждой угловой компоненты. Повторный расчёт этих величин и давал
    основную стоимость градиента.
    """
    powers_a = cartesian_powers(shell_a.angular_momentum)
    powers_b = cartesian_powers(shell_b.angular_momentum)
    powers_c = cartesian_powers(shell_c.angular_momentum)
    powers_d = cartesian_powers(shell_d.angular_momentum)
    scales = (
        shell_a.component_scales,
        shell_b.component_scales,
        shell_c.component_scales,
        shell_d.component_scales,
    )
    counts = [
        len(shell_a.exponents),
        len(shell_b.exponents),
        len(shell_c.exponents),
        len(shell_d.exponents),
    ]
    a = np.repeat(np.asarray(shell_a.exponents, dtype=float), counts[1] * counts[2] * counts[3])
    b = np.repeat(
        np.tile(np.asarray(shell_b.exponents, dtype=float), counts[0]), counts[2] * counts[3]
    )
    c = np.repeat(
        np.tile(np.asarray(shell_c.exponents, dtype=float), counts[0] * counts[1]), counts[3]
    )
    d = np.tile(np.asarray(shell_d.exponents, dtype=float), counts[0] * counts[1] * counts[2])
    coefficients = np.einsum(
        "i,j,k,l->ijkl",
        np.asarray(shell_a.coefficients, dtype=float),
        np.asarray(shell_b.coefficients, dtype=float),
        np.asarray(shell_c.coefficients, dtype=float),
        np.asarray(shell_d.coefficients, dtype=float),
    ).reshape(-1)

    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    pqx = (a * center_a[0] + b * center_b[0]) / p - (c * center_c[0] + d * center_d[0]) / q
    pqy = (a * center_a[1] + b * center_b[1]) / p - (c * center_c[1] + d * center_d[1]) / q
    pqz = (a * center_a[2] + b * center_b[2]) / p - (c * center_c[2] + d * center_d[2]) / q
    x = alpha * (pqx * pqx + pqy * pqy + pqz * pqz)
    prefactor = 2.0 * PI_5_2 / (p * q * np.sqrt(p + q))

    qx = center_a[0] - center_b[0]
    qy = center_a[1] - center_b[1]
    qz = center_a[2] - center_b[2]
    rx = center_c[0] - center_d[0]
    ry = center_c[1] - center_d[1]
    rz = center_c[2] - center_d[2]

    highest = (
        shell_a.angular_momentum
        + shell_b.angular_momentum
        + 1
        + shell_c.angular_momentum
        + shell_d.angular_momentum
    )
    coulomb = _coulomb_table(highest, alpha, pqx, pqy, pqz, x)

    block = np.zeros((len(powers_a), len(powers_b), len(powers_c), len(powers_d)))
    for ib, pb in enumerate(powers_b):
        for ic, pc in enumerate(powers_c):
            for idx, pd in enumerate(powers_d):
                # Кет-сторона от угловых моментов бра-функции не зависит,
                # поэтому её коэффициенты считаются один раз на (ib, ic, idx).
                ket = _hermite_product_arrays(pc, pd, rx, ry, rz, c, d, alternating=True)
                if not ket:
                    continue
                for ia, pa in enumerate(powers_a):
                    raised = list(pa)
                    raised[axis] += 1
                    bra = _hermite_product_arrays(
                        (raised[0], raised[1], raised[2]), pb, qx, qy, qz, a, b
                    )
                    values = 2.0 * a * _contract(bra, ket, coulomb)
                    if pa[axis] > 0:
                        lowered = list(pa)
                        lowered[axis] -= 1
                        bra_lower = _hermite_product_arrays(
                            (lowered[0], lowered[1], lowered[2]), pb, qx, qy, qz, a, b
                        )
                        values = values - float(pa[axis]) * _contract(bra_lower, ket, coulomb)
                    block[ia, ib, ic, idx] = float(coefficients @ (prefactor * values)) * (
                        scales[0][ia] * scales[1][ib] * scales[2][ic] * scales[3][idx]
                    )
    return block


def _quartet_derivative_block_scalar(
    axis: int,
    shell_a: Shell,
    center_a: Point,
    shell_b: Shell,
    center_b: Point,
    shell_c: Shell,
    center_c: Point,
    shell_d: Shell,
    center_d: Point,
) -> np.ndarray:
    """Блок производной ERI, считанный «в лоб» по примитивным квартетам.

    Рабочий путь — векторизованный :func:`_quartet_derivative_block`; эта
    функция оставлена как читаемая спецификация формул и как оракул для
    теста на совпадение двух путей.
    """
    powers_a = cartesian_powers(shell_a.angular_momentum)
    powers_b = cartesian_powers(shell_b.angular_momentum)
    powers_c = cartesian_powers(shell_c.angular_momentum)
    powers_d = cartesian_powers(shell_d.angular_momentum)
    scales = (
        shell_a.component_scales,
        shell_b.component_scales,
        shell_c.component_scales,
        shell_d.component_scales,
    )
    block = np.zeros((len(powers_a), len(powers_b), len(powers_c), len(powers_d)))
    for ia, pa in enumerate(powers_a):
        for ib, pb in enumerate(powers_b):
            for ic, pc in enumerate(powers_c):
                for idx, pd in enumerate(powers_d):
                    total = 0.0
                    for alpha, ca in zip(shell_a.exponents, shell_a.coefficients, strict=True):
                        for beta, cb in zip(shell_b.exponents, shell_b.coefficients, strict=True):
                            for gamma, cc in zip(
                                shell_c.exponents, shell_c.coefficients, strict=True
                            ):
                                for delta, cd in zip(
                                    shell_d.exponents, shell_d.coefficients, strict=True
                                ):
                                    total += (
                                        ca
                                        * cb
                                        * cc
                                        * cd
                                        * _eri_derivative_primitive(
                                            axis,
                                            alpha,
                                            pa,
                                            center_a,
                                            beta,
                                            pb,
                                            center_b,
                                            gamma,
                                            pc,
                                            center_c,
                                            delta,
                                            pd,
                                            center_d,
                                        )
                                    )
                    block[ia, ib, ic, idx] = total * (
                        scales[0][ia] * scales[1][ib] * scales[2][ic] * scales[3][idx]
                    )
    return block


def _place_quartet_derivative(
    tensor: np.ndarray, basis: BasisSet, i: int, j: int, k: int, m: int, block: np.ndarray
) -> None:
    """Размещает блок производной ERI, используя только симметрию λ↔σ."""
    oi, oj, ok, om = (_shell_offset(basis, index) for index in (i, j, k, m))
    si = slice(oi, oi + basis.shells[i].n_cartesian)
    sj = slice(oj, oj + basis.shells[j].n_cartesian)
    sk = slice(ok, ok + basis.shells[k].n_cartesian)
    sm = slice(om, om + basis.shells[m].n_cartesian)
    tensor[si, sj, sk, sm] = block
    tensor[si, sj, sm, sk] = block.transpose(0, 1, 3, 2)


# --------------------------------------------------------------------------- #
# Пакетная сборка двухэлектронных интегралов
# --------------------------------------------------------------------------- #
# Стоимость сборки ERI в этой реализации определяет не арифметика, а число
# вызовов NumPy: на воде/cc-pVDZ это 83 тысячи вызовов `_hermite_1d` на 3081
# квартет оболочек — около 600 мкс на квартет при 25 базисных функциях, тогда
# как сама арифметика занимает единицы наносекунд на элемент. Пачка квартетов
# одного класса объединяет их примитивные квартеты в один массив: объём
# арифметики тот же, но накладные расходы интерпретатора делятся на размер
# пачки.
#
# Класс квартета — угловые моменты четырёх оболочек **и длины их сжатий**.
# Длины входят в ключ потому, что примитивная ось пачки прямоугольная:
# квартеты одного класса с разным числом примитивов пришлось бы добивать
# фиктивными экспонентами с нулевым весом, то есть считать впустую. В реальных
# базисах различных длин единицы (6-31G: 6 и 3), поэтому пачки остаются
# большими.


#: Квартет оболочек: индексы ``(i, j, k, m)``.
Quartet = tuple[int, int, int, int]

#: Ключ класса квартета: ``(l_i, l_j, l_k, l_m, n_i, n_j, n_k, n_m)``.
ClassKey = tuple[int, int, int, int, int, int, int, int]

#: Целевой размер рабочего массива, в примитивных квартетах.
#:
#: Выбран измерением на воде/cc-pVDZ и бензоле/6-31G: дальше рост пачки
#: время не улучшает — сборка упирается в пропускную способность памяти.
_TARGET_BATCH_ELEMENTS: Final = 1 << 16

#: Предел памяти на таблицу ``R^n_{tuv}``, байт.
#:
#: Таблица — это ``C(limit+4, 4)`` массивов рабочего размера: для ``s``-оболочек
#: один, для квартета ``d``-оболочек 495. Ограничивать только число элементов
#: значило бы занимать сотни мегабайт на классах с высоким угловым моментом.
_TARGET_TABLE_BYTES: Final = 32 << 20


def _table_entries(limit: int) -> int:
    """Число ячеек ``(t, u, v, n)`` с ``t+u+v+n ≤ limit``: ``C(limit+4, 4)``."""
    return (limit + 1) * (limit + 2) * (limit + 3) * (limit + 4) // 24


def _shell_offsets(basis: BasisSet) -> np.ndarray:
    """Смещения оболочек в матрице базисных функций (префиксная сумма).

    :func:`_shell_offset` суммирует префикс на каждый вызов; в пакетном пути
    смещений нужно по четыре на пачку, поэтому они считаются один раз.
    """
    sizes = np.array([shell.n_cartesian for shell in basis.shells], dtype=int)
    return np.concatenate((np.zeros(1, dtype=int), np.cumsum(sizes)))


def _unique_quartets(n_shells: int) -> Iterator[Quartet]:
    """Уникальные квартеты для энергии: ``(ij) ≥ (kl)`` в порядке пар.

    Условие оставляет ровно одного представителя из восьми перестановок
    тензора; остальные заполняются при размещении блока.
    """
    for i in range(n_shells):
        for j in range(i + 1):
            left = i * (i + 1) + j
            for k in range(n_shells):
                for m in range(k + 1):
                    if left >= k * (k + 1) + m:
                        yield (i, j, k, m)


def _bra_quartets(n_shells: int) -> Iterator[Quartet]:
    """Квартеты для производной по центру бра: пара ``(ij)`` полная.

    У производной по центру μ восьмикратная симметрия тензора не выполняется,
    поэтому бра-пара перебирается целиком; остаётся только ``(kl) = (lk)``.
    """
    for i in range(n_shells):
        for j in range(n_shells):
            for k in range(n_shells):
                for m in range(k + 1):
                    yield (i, j, k, m)


def _class_key(shells: Sequence[Shell], quartet: Quartet) -> ClassKey:
    i, j, k, m = quartet
    a, b, c, d = shells[i], shells[j], shells[k], shells[m]
    return (
        a.angular_momentum,
        b.angular_momentum,
        c.angular_momentum,
        d.angular_momentum,
        len(a.exponents),
        len(b.exponents),
        len(c.exponents),
        len(d.exponents),
    )


def _group_by_class(
    shells: Sequence[Shell], quartets: Iterable[Quartet]
) -> dict[ClassKey, list[Quartet]]:
    """Раскладывает квартеты по классам: внутри класса форма блока одна."""
    groups: dict[ClassKey, list[Quartet]] = {}
    for quartet in quartets:
        groups.setdefault(_class_key(shells, quartet), []).append(quartet)
    return groups


@dataclass(frozen=True, slots=True)
class _ClassGeometry:
    """Всё, что у квартетов одного класса совпадает.

    Attributes:
        powers: декартовы степени по четырём сторонам.
        scales: нормировочные множители компонент (``component_scales``).
        shape: форма блока ``( |A|, |B|, |C|, |D| )``.
        counts: длины сжатий оболочек.
        angular_budget: максимальная сумма ``t+u+v`` в таблице ``R``.
    """

    powers: tuple[tuple[Powers, ...], tuple[Powers, ...], tuple[Powers, ...], tuple[Powers, ...]]
    scales: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    shape: tuple[int, int, int, int]
    counts: tuple[int, int, int, int]
    angular_budget: int


def _class_geometry(shells: Sequence[Shell], quartet: Quartet) -> _ClassGeometry:
    """Собирает геометрию класса по любому его квартету (они одинаковы)."""
    parts = [shells[index] for index in quartet]
    powers = tuple(cartesian_powers(shell.angular_momentum) for shell in parts)
    return _ClassGeometry(
        powers=(powers[0], powers[1], powers[2], powers[3]),
        scales=(
            parts[0].component_scales,
            parts[1].component_scales,
            parts[2].component_scales,
            parts[3].component_scales,
        ),
        shape=(len(powers[0]), len(powers[1]), len(powers[2]), len(powers[3])),
        counts=(
            len(parts[0].exponents),
            len(parts[1].exponents),
            len(parts[2].exponents),
            len(parts[3].exponents),
        ),
        angular_budget=sum(shell.angular_momentum for shell in parts),
    )


def _batch_step(geometry: _ClassGeometry, angular_budget: int) -> int:
    """Число квартетов в пачке.

    Два ограничения: целевой размер рабочего массива и память на таблицу
    ``R^n_{tuv}``, которая растёт как ``C(limit+4, 4)``. Второе срабатывает на
    классах с высоким угловым моментом, где примитивов мало, а ячеек таблицы
    сотни — без него пачка ``(d,d|d,d)`` заняла бы сотни мегабайт.
    """
    n, a, b, c = geometry.counts
    primitives = n * a * b * c
    by_elements = _TARGET_BATCH_ELEMENTS // primitives
    by_memory = _TARGET_TABLE_BYTES // (8 * _table_entries(angular_budget) * primitives)
    return max(1, min(by_elements, by_memory))


@dataclass(frozen=True, slots=True)
class _BatchArrays:
    """Массивы примитивных квартетов пачки одного класса.

    Полная форма — ``(N, n_a, n_b, n_c, n_d)``. Экспоненты хранятся
    «свёрнутыми» по своей оси — ``(N, n_a, 1, 1, 1)`` и т. д. — и расширяются
    арифметикой NumPy. Это не только экономия памяти: эрмитовы коэффициенты
    бра-стороны тогда живут на ``(N, n_a, n_b, 1, 1)``, то есть рекурсия
    Хельгакера не повторяется ``n_c·n_d`` раз, как в поквартиетном пути.

    Attributes:
        a: экспоненты бра-функции ``a``, форма ``(N, n_a, 1, 1, 1)``.
        b: экспоненты бра-функции ``b``, форма ``(N, 1, n_b, 1, 1)``.
        c: экспоненты кет-функции ``c``, форма ``(N, 1, 1, n_c, 1)``.
        d: экспоненты кет-функции ``d``, форма ``(N, 1, 1, 1, n_d)``.
        contraction: коэффициенты сжатия по сторонам, каждый ``(N, n_side)``.
        prefactor: ``2π^{5/2}/(pq√(p+q))``, полная форма.
        coulomb: таблица ``R^n_{tuv}``, полная форма.
        q: векторы ``A − B`` по квартетам, форма ``(N, 1, 1, 1, 1)``.
        r: векторы ``C − D`` по квартетам, форма ``(N, 1, 1, 1, 1)``.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    contraction: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    prefactor: np.ndarray
    coulomb: dict[tuple[int, int, int, int], np.ndarray]
    q: tuple[np.ndarray, np.ndarray, np.ndarray]
    r: tuple[np.ndarray, np.ndarray, np.ndarray]


def _axis_vectors(values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(N, 3)`` → три массива формы ``(N, 1, 1, 1, 1)`` по декартовым осям."""
    return (
        values[:, 0].reshape(n, 1, 1, 1, 1),
        values[:, 1].reshape(n, 1, 1, 1, 1),
        values[:, 2].reshape(n, 1, 1, 1, 1),
    )


def _side_positions(
    centers: Sequence[Point], quartets: Sequence[Quartet], side: int, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Координаты центра ``side``-й оболочки каждого квартета пачки."""
    stacked = np.stack([np.asarray(centers[quartet[side]], dtype=float) for quartet in quartets])
    return _axis_vectors(stacked, n)


def _batch_arrays(
    shells: Sequence[Shell],
    centers: Sequence[Point],
    quartets: Sequence[Quartet],
    *,
    angular_budget: int,
) -> _BatchArrays:
    """Раскладывает пачку квартетов по примитивным квартетам сразу.

    Порядок примитивов внутри квартета — прямое произведение ``(α, β, γ, δ)``
    с медленнее всего меняющимся ``α``, то есть тот же, что в
    :func:`_quartet_arrays`; меняется только добавленный первый индекс пачки.

    Args:
        shells: оболочки базиса.
        centers: центры оболочек.
        quartets: квартеты пачки; все одного класса.
        angular_budget: максимальная сумма ``t+u+v`` в таблице ``R`` — сумма
            угловых моментов для энергии и на единицу больше для производной.
    """
    n = len(quartets)
    exponents = [
        np.stack([np.asarray(shells[quartet[side]].exponents, dtype=float) for quartet in quartets])
        for side in range(4)
    ]
    contraction = [
        np.stack(
            [np.asarray(shells[quartet[side]].coefficients, dtype=float) for quartet in quartets]
        )
        for side in range(4)
    ]
    a = exponents[0].reshape(n, exponents[0].shape[1], 1, 1, 1)
    b = exponents[1].reshape(n, 1, exponents[1].shape[1], 1, 1)
    c = exponents[2].reshape(n, 1, 1, exponents[2].shape[1], 1)
    d = exponents[3].reshape(n, 1, 1, 1, exponents[3].shape[1])

    position_a = _side_positions(centers, quartets, 0, n)
    position_b = _side_positions(centers, quartets, 1, n)
    position_c = _side_positions(centers, quartets, 2, n)
    position_d = _side_positions(centers, quartets, 3, n)

    p = a + b
    s = c + d
    total = p + s
    alpha = p * s / total
    pqx = (a * position_a[0] + b * position_b[0]) / p - (c * position_c[0] + d * position_d[0]) / s
    pqy = (a * position_a[1] + b * position_b[1]) / p - (c * position_c[1] + d * position_d[1]) / s
    pqz = (a * position_a[2] + b * position_b[2]) / p - (c * position_c[2] + d * position_d[2]) / s
    x = alpha * (pqx * pqx + pqy * pqy + pqz * pqz)

    return _BatchArrays(
        a=a,
        b=b,
        c=c,
        d=d,
        contraction=(contraction[0], contraction[1], contraction[2], contraction[3]),
        prefactor=2.0 * PI_5_2 / (p * s * np.sqrt(total)),
        coulomb=_coulomb_table(angular_budget, alpha, pqx, pqy, pqz, x),
        q=(
            (position_a[0] - position_b[0]).reshape(n, 1, 1, 1, 1),
            (position_a[1] - position_b[1]).reshape(n, 1, 1, 1, 1),
            (position_a[2] - position_b[2]).reshape(n, 1, 1, 1, 1),
        ),
        r=(
            (position_c[0] - position_d[0]).reshape(n, 1, 1, 1, 1),
            (position_c[1] - position_d[1]).reshape(n, 1, 1, 1, 1),
            (position_c[2] - position_d[2]).reshape(n, 1, 1, 1, 1),
        ),
    )


def _contract_primitives(arrays: _BatchArrays, values: np.ndarray) -> np.ndarray:
    r"""Свёртка примитивной оси с коэффициентами сжатия: ``(N, …) → (N,)``.

    Коэффициенты хранятся по сторонам, а не внешним произведением, поэтому
    свёртка идёт четырьмя проходами и массив ``c_a c_b c_c c_d`` не строится.
    Результат совпадает с ``coefficients @ (prefactor · values)`` с точностью
    до порядка суммирования (проверено тестом, расхождение ≤ 1e-13).
    """
    ca, cb, cc, cd = arrays.contraction
    x = arrays.prefactor * values
    x = np.einsum("na,nabcd->nbcd", ca, x)
    x = np.einsum("nb,nbcd->ncd", cb, x)
    x = np.einsum("nc,ncd->nd", cc, x)
    return np.asarray(np.einsum("nd,nd->n", cd, x))


def _batch_eri_blocks(arrays: _BatchArrays, geometry: _ClassGeometry) -> np.ndarray:
    """Блоки ``(ab|cd)`` всей пачки: форма ``(N, |A|, |B|, |C|, |D|)``.

    Формула та же, что в :func:`_quartet_block`: кет-коэффициенты зависят
    только от пары ``(C, D)``, бра-коэффициенты — только от ``(A, B)``, а
    таблица ``R`` одна на пачку.
    """
    powers_a, powers_b, powers_c, powers_d = geometry.powers
    scales_a, scales_b, scales_c, scales_d = geometry.scales
    n = arrays.contraction[0].shape[0]
    blocks = np.zeros((n, *geometry.shape))

    ket_cache: dict[tuple[int, int, int], list[np.ndarray]] = {}
    ket_table: dict[tuple[int, int], dict[Powers, np.ndarray]] = {}
    for ic, pc in enumerate(powers_c):
        for idx, pd in enumerate(powers_d):
            ket_table[(ic, idx)] = _hermite_product_arrays(
                pc,
                pd,
                arrays.r[0],
                arrays.r[1],
                arrays.r[2],
                arrays.c,
                arrays.d,
                alternating=True,
                cache=ket_cache,
            )
    # Если кет-сторона пуста целиком, бра считать незачем.
    if not any(ket_table.values()):
        return blocks
    bra_cache: dict[tuple[int, int, int], list[np.ndarray]] = {}
    bra_table: dict[tuple[int, int], dict[Powers, np.ndarray]] = {}
    for ia, pa in enumerate(powers_a):
        for ib, pb in enumerate(powers_b):
            bra_table[(ia, ib)] = _hermite_product_arrays(
                pa,
                pb,
                arrays.q[0],
                arrays.q[1],
                arrays.q[2],
                arrays.a,
                arrays.b,
                cache=bra_cache,
            )

    for (ic, idx), ket in ket_table.items():
        if not ket:
            continue
        for (ia, ib), bra in bra_table.items():
            values = _contract(bra, ket, arrays.coulomb)
            blocks[:, ia, ib, ic, idx] = _contract_primitives(arrays, values) * (
                scales_a[ia] * scales_b[ib] * scales_c[ic] * scales_d[idx]
            )
    return blocks


def _batch_eri_derivative_blocks(
    axis: int, arrays: _BatchArrays, geometry: _ClassGeometry
) -> np.ndarray:
    r"""Блоки ``∂(ab|cd)/∂A_x`` всей пачки: ``(N, |A|, |B|, |C|, |D|)``.

    Та же формула, что в :func:`_quartet_derivative_block`::

        ∂(ab|cd)/∂A_x = 2a·(a+e_x, b|c,d) − l_a,x·(a−e_x, b|c,d)
    """
    powers_a, powers_b, powers_c, powers_d = geometry.powers
    scales_a, scales_b, scales_c, scales_d = geometry.scales
    n = arrays.contraction[0].shape[0]
    blocks = np.zeros((n, *geometry.shape))
    bra_cache: dict[tuple[int, int, int], list[np.ndarray]] = {}
    ket_cache: dict[tuple[int, int, int], list[np.ndarray]] = {}

    for ib, pb in enumerate(powers_b):
        for ic, pc in enumerate(powers_c):
            for idx, pd in enumerate(powers_d):
                ket = _hermite_product_arrays(
                    pc,
                    pd,
                    arrays.r[0],
                    arrays.r[1],
                    arrays.r[2],
                    arrays.c,
                    arrays.d,
                    alternating=True,
                    cache=ket_cache,
                )
                if not ket:
                    continue
                for ia, pa in enumerate(powers_a):
                    raised = list(pa)
                    raised[axis] += 1
                    bra = _hermite_product_arrays(
                        (raised[0], raised[1], raised[2]),
                        pb,
                        arrays.q[0],
                        arrays.q[1],
                        arrays.q[2],
                        arrays.a,
                        arrays.b,
                        cache=bra_cache,
                    )
                    values = 2.0 * arrays.a * _contract(bra, ket, arrays.coulomb)
                    if pa[axis] > 0:
                        lowered = list(pa)
                        lowered[axis] -= 1
                        bra_lower = _hermite_product_arrays(
                            (lowered[0], lowered[1], lowered[2]),
                            pb,
                            arrays.q[0],
                            arrays.q[1],
                            arrays.q[2],
                            arrays.a,
                            arrays.b,
                            cache=bra_cache,
                        )
                        values = values - float(pa[axis]) * _contract(
                            bra_lower, ket, arrays.coulomb
                        )
                    blocks[:, ia, ib, ic, idx] = _contract_primitives(arrays, values) * (
                        scales_a[ia] * scales_b[ib] * scales_c[ic] * scales_d[idx]
                    )
    return blocks


#: Восемь перестановок тензора ``(μν|λσ)``: какие стороны стоят в позициях
#: тензора. Например, ``(2, 3, 0, 1)`` — это ``(λσ|μν)``.
_ERI_PERMUTATIONS: Final[tuple[tuple[int, int, int, int], ...]] = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


#: У производной по центру μ восьмикратная симметрия тензора не выполняется:
#: остаётся только ``(μν|λσ) = (μν|σλ)``, то есть перестановка кет-сторон.
_ERI_DERIVATIVE_PERMUTATIONS: Final[tuple[tuple[int, int, int, int], ...]] = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
)


def _component_range(size: int, position: int) -> np.ndarray:
    """Смещения компонент оболочки вдоль оси результата ``position + 1``."""
    shape = [1, 1, 1, 1, 1]
    shape[position + 1] = size
    return np.arange(size, dtype=int).reshape(tuple(shape))


def _write_permutations(
    tensor: np.ndarray,
    offsets: np.ndarray,
    quartets: Sequence[Quartet],
    geometry: _ClassGeometry,
    blocks: np.ndarray,
    permutations: Sequence[tuple[int, int, int, int]],
) -> None:
    """Записывает пачку блоков в тензор по заданным перестановкам сторон.

    Форма левой части присваивания у NumPy — результат обычного broadcasting
    индексных массивов: ось, по которой меняются компоненты оболочки, стоит там,
    где её поставили при построении индекса. Поэтому ``order`` задаёт и индексы,
    и транспозицию блока: при ``order = (1, 0, 2, 3)`` в позицию 0 тензора
    попадает сторона ``b``, и первая после батчевой ось блока тоже должна
    оказаться осью ``b`` — отсюда ``(0, 2, 1, 3, 4)``.
    """
    n = len(quartets)
    counts = geometry.shape
    starts = [
        np.array([offsets[quartet[side]] for quartet in quartets], dtype=int).reshape(n, 1, 1, 1, 1)
        for side in range(4)
    ]
    for order in permutations:
        index = tuple(
            starts[side] + _component_range(counts[side], position)
            for position, side in enumerate(order)
        )
        tensor[index] = blocks.transpose(
            (0, order[0] + 1, order[1] + 1, order[2] + 1, order[3] + 1)
        )


def _place_quartet_batch(
    tensor: np.ndarray,
    offsets: np.ndarray,
    quartets: Sequence[Quartet],
    geometry: _ClassGeometry,
    blocks: np.ndarray,
) -> None:
    """Записывает пачку блоков в тензор со всеми восемью перестановками.

    Индексы строятся сразу на всю пачку, поэтому вместо восьми записей на
    квартет выполняется восемь записей на пачку. При 400 квартетах в пачке
    это на три порядка меньше обращений к NumPy — после ускорения ядра
    размещение само стало бы узким местом.
    """
    _write_permutations(tensor, offsets, quartets, geometry, blocks, _ERI_PERMUTATIONS)


def _place_quartet_derivative_batch(
    tensor: np.ndarray,
    offsets: np.ndarray,
    quartets: Sequence[Quartet],
    geometry: _ClassGeometry,
    blocks: np.ndarray,
) -> None:
    """Пакетный аналог :func:`_place_quartet_derivative`: только ``λ ↔ σ``."""
    _write_permutations(tensor, offsets, quartets, geometry, blocks, _ERI_DERIVATIVE_PERMUTATIONS)


def _build_electron_repulsion_quartetwise(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """Тензор ERI, собранный по одному квартету оболочек за раз.

    Эталон для теста, а не рабочий путь: :func:`build_electron_repulsion`
    считает те же числа пачками. Сравнение идёт по всему тензору, а не по
    отдельным блокам, поэтому ловит и ошибки размещения блоков.
    """
    centers = _shell_centers(basis, molecule)
    n_shells = len(basis.shells)
    tensor = np.zeros((basis.n_functions,) * 4)
    for i, j, k, m in _unique_quartets(n_shells):
        block = _quartet_block(
            basis.shells[i],
            centers[i],
            basis.shells[j],
            centers[j],
            basis.shells[k],
            centers[k],
            basis.shells[m],
            centers[m],
        )
        _place_quartet(tensor, basis, i, j, k, m, block)
    return tensor


def _build_electron_repulsion_derivative_quartetwise(
    basis: BasisSet, molecule: Molecule, axis: int
) -> np.ndarray:
    """Тензор ``∂(μν|λσ)/∂A_x`` по одному квартету за раз — эталон для теста."""
    centers = _shell_centers(basis, molecule)
    n_shells = len(basis.shells)
    tensor = np.zeros((basis.n_functions,) * 4)
    for i, j, k, m in _bra_quartets(n_shells):
        block = _quartet_derivative_block(
            axis,
            basis.shells[i],
            centers[i],
            basis.shells[j],
            centers[j],
            basis.shells[k],
            centers[k],
            basis.shells[m],
            centers[m],
        )
        _place_quartet_derivative(tensor, basis, i, j, k, m, block)
    return tensor


def build_dipole_integrals(
    basis: BasisSet, molecule: Molecule
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Интегралы ``⟨μ|x|ν⟩``, ``⟨μ|y|ν⟩``, ``⟨μ|z|ν⟩`` в атомных единицах.

    Начало координат — общая точка отсчёта (0, 0, 0), поэтому ядерный вклад в
    диполь считается от тех же координат и полный диполь не зависит от выбора
    начала координат для нейтральной системы.
    """
    centers = _shell_centers(basis, molecule)
    size = basis.n_functions
    matrices: list[np.ndarray] = []
    for axis in range(3):

        def kernel(
            a: float,
            la: Powers,
            center_a: Point,
            b: float,
            lb: Powers,
            center_b: Point,
            _axis: int = axis,
        ) -> float:
            return _multipole_primitive(a, la, center_a, b, lb, center_b, _axis)

        matrix = np.zeros((size, size))
        for i, shell_a in enumerate(basis.shells):
            for j, shell_b in enumerate(basis.shells[: i + 1]):
                block = _pair_block(shell_a, centers[i], shell_b, centers[j], kernel)
                _place_block(matrix, basis, i, j, block)
        matrices.append(matrix)
    return (matrices[0], matrices[1], matrices[2])


def clear_caches() -> None:
    """Сбрасывает кэш эрмитовых коэффициентов."""
    _hermite_coefficient.cache_clear()
