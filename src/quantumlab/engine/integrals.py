"""Одно- и двухэлектронные интегралы по декартовым гауссианам.

Алгоритм — McMurchie–Davidson (соотношения из Helgaker, Jørgensen, Olsen,
*Molecular Electronic-Structure Theory*, гл. 9): коэффициенты Эрмитовых
разложений ``E`` и эрмитовы кулоновские интегралы ``R`` вычисляются
рекуррентно, что даёт единый код для любого углового момента.

Сложность без скрининга: одноэлектронные — O(N²), двухэлектронные — O(N⁴) по
числу оболочек. Это **референсная** реализация (ADR-002): плотные массивы,
один поток, без отсечек — её задача быть эталоном корректности.

Единицы — атомные (бор, хартри).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import BasisSet, Shell, cartesian_powers
from quantumlab.engine.constants import PI_5_2, angstrom_to_bohr

#: Ниже этого аргумента функция Бойса считается рядом Тейлора.
_BOYS_SERIES_LIMIT = 6.0

#: Число членов ряда (с запасом для x ≤ 6).
_BOYS_SERIES_TERMS = 25

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


def _boys(n: int, x: float) -> float:
    """Функция Бойса ``F_n(x) = ∫₀¹ t^{2n} e^{-x t²} dt``.

    Для малых ``x`` — ряд Тейлора (устойчив, пока члены не начинают
    сокращаться), для больших — рекурсия вверх от
    ``F_0 = ½√(π/x)·erf(√x)``, которая устойчива при ``x ≫ n``.
    """
    if x < 1e-12:
        return 1.0 / (2 * n + 1)
    if x < _BOYS_SERIES_LIMIT:
        total = 0.0
        term = 1.0
        for k in range(_BOYS_SERIES_TERMS):
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
    """
    centers = _shell_centers(basis, molecule)
    n_shells = len(basis.shells)
    tensor = np.zeros((basis.n_functions,) * 4)

    for i in range(n_shells):
        for j in range(i + 1):
            pair_left = i * (i + 1) + j
            for k in range(n_shells):
                for m in range(k + 1):
                    if pair_left < k * (k + 1) + m:
                        continue
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
