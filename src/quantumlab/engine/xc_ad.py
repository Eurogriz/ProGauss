"""Автоматическое дифференцирование первого порядка для XC-функционалов.

Спин-поляризованные meta-GGA (M06, M06-2X, TPSSh в UKS, ωB97X) — это длинные
формулы от семи переменных ``(ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β)``. Выводить
их производные вручную — значит нести десятки страниц алгебры, каждая строка
которой может скрыть ошибку, невидимую по энергии (так уже было с цепным
правилом spin-GGA, см. :mod:`quantumlab.engine.xc_spin_cores`). Здесь формулы
записываются один раз — в виде кода, повторяющего формулы LibXC, — а
производные даёт прямое дифференцирование (forward mode).

Для SCF и градиента нужны только первые производные ``∂E_V/∂(переменная)``,
поэтому хранятся именно они. Производные — **разреженные**: словарь
``{индекс переменной: массив}``; обменная часть одного спин-канала зависит
лишь от трёх переменных, и считать для неё семь частных производных было бы
расточительно.

Модуль ничего не знает про конкретные функционалы: только арифметика и
элементарные функции.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

Operand: TypeAlias = "Dual | float | np.ndarray"

_TWO_OVER_SQRT_PI: float = 2.0 / math.sqrt(math.pi)
_erf_scalar = np.frompyfunc(math.erf, 1, 1)


def _erf(x: np.ndarray) -> np.ndarray:
    """``erf`` по массиву (``math.erf`` — точный; SciPy проектом не требуется)."""
    return np.asarray(_erf_scalar(x), dtype=float)


class Dual:
    """Значение и его частные производные по независимым переменным.

    ``d`` — разреженный словарь ``{индекс: массив}``: отсутствующий ключ значит
    «производная равна нулю». Все арифметические операции возвращают новый
    объект и не изменяют операндов.
    """

    __slots__ = ("d", "v")
    #: Чтобы ``ndarray * Dual`` вызывал ``Dual.__rmul__``, а не поэлементное умножение.
    __array_ufunc__ = None

    def __init__(
        self, value: np.ndarray | float, derivatives: dict[int, np.ndarray] | None = None
    ) -> None:
        """Хранит значение и словарь производных."""
        self.v = np.asarray(value, dtype=float)
        self.d: dict[int, np.ndarray] = {} if derivatives is None else derivatives

    @staticmethod
    def variable(index: int, value: np.ndarray) -> Dual:
        """Независимая переменная с номером ``index``."""
        array = np.asarray(value, dtype=float)
        return Dual(array, {index: np.ones_like(array)})

    # -- арифметика ---------------------------------------------------------
    def __add__(self, other: Operand) -> Dual:
        """Сумма."""
        if not isinstance(other, Dual):
            return Dual(self.v + other, self.d)
        out: dict[int, np.ndarray] = {}
        for key in self.d.keys() | other.d.keys():
            a = self.d.get(key)
            b = other.d.get(key)
            if a is not None and b is not None:
                out[key] = a + b
            elif a is not None:
                out[key] = a
            else:
                assert b is not None
                out[key] = b
        return Dual(self.v + other.v, out)

    __radd__ = __add__

    def __neg__(self) -> Dual:
        """Смена знака."""
        return Dual(-self.v, {k: -a for k, a in self.d.items()})

    def __sub__(self, other: Operand) -> Dual:
        """Разность."""
        if not isinstance(other, Dual):
            return Dual(self.v - other, self.d)
        return self + (-other)

    def __rsub__(self, other: Operand) -> Dual:
        """Разность с константой слева."""
        return (-self) + other

    def __mul__(self, other: Operand) -> Dual:
        """Произведение."""
        if not isinstance(other, Dual):
            return Dual(self.v * other, {k: a * other for k, a in self.d.items()})
        out: dict[int, np.ndarray] = {}
        for key in self.d.keys() | other.d.keys():
            a = self.d.get(key)
            b = other.d.get(key)
            if a is not None and b is not None:
                out[key] = a * other.v + self.v * b
            elif a is not None:
                out[key] = a * other.v
            else:
                assert b is not None
                out[key] = self.v * b
        return Dual(self.v * other.v, out)

    __rmul__ = __mul__

    def __truediv__(self, other: Operand) -> Dual:
        """Частное."""
        if not isinstance(other, Dual):
            return Dual(self.v / other, {k: a / other for k, a in self.d.items()})
        return self * other.reciprocal()

    def __rtruediv__(self, other: Operand) -> Dual:
        """Частное с константой в числителе."""
        return self.reciprocal() * other

    def reciprocal(self) -> Dual:
        """``1/x``."""
        inv = 1.0 / self.v
        slope = -inv * inv
        return Dual(inv, {k: a * slope for k, a in self.d.items()})

    def __pow__(self, exponent: float) -> Dual:
        """Степень с постоянным показателем (основание положительно)."""
        value = self.v**exponent
        slope = exponent * self.v ** (exponent - 1.0)
        return Dual(value, {k: a * slope for k, a in self.d.items()})

    # -- служебное ------------------------------------------------------------
    def chain(self, value: np.ndarray, slope: np.ndarray) -> Dual:
        """``f(self)`` по известным значению и производной ``f'``."""
        return Dual(value, {k: a * slope for k, a in self.d.items()})

    def masked(self, mask: np.ndarray) -> Dual:
        """Обнуляет значение и производные там, где ``mask`` ложно."""
        return Dual(
            np.where(mask, self.v, 0.0),
            {k: np.where(mask, a, 0.0) for k, a in self.d.items()},
        )

    def finite(self) -> Dual:
        """Заменяет нечисловые производные нулями (хвосты сетки)."""
        return Dual(self.v, {k: np.where(np.isfinite(a), a, 0.0) for k, a in self.d.items()})


def as_dual(x: Operand) -> Dual:
    """Константа или массив как ``Dual`` без производных."""
    return x if isinstance(x, Dual) else Dual(np.asarray(x, dtype=float))


def sqrt(x: Dual) -> Dual:
    """Квадратный корень."""
    root = np.sqrt(x.v)
    return x.chain(root, 0.5 / root)


def exp(x: Dual) -> Dual:
    """Экспонента."""
    value = np.exp(x.v)
    return x.chain(value, value)


def log(x: Dual) -> Dual:
    """Натуральный логарифм."""
    return x.chain(np.log(x.v), 1.0 / x.v)


def log1p(x: Dual) -> Dual:
    """``ln(1 + x)`` без потери точности при малых ``x``."""
    return x.chain(np.log1p(x.v), 1.0 / (1.0 + x.v))


def erf(x: Dual) -> Dual:
    """Функция ошибок."""
    return x.chain(_erf(x.v), _TWO_OVER_SQRT_PI * np.exp(-(x.v * x.v)))


def where(condition: np.ndarray, a: Operand, b: Operand) -> Dual:
    """Поэлементный выбор ``a``/``b`` (как ``np.where``) вместе с производными."""
    left, right = as_dual(a), as_dual(b)
    out: dict[int, np.ndarray] = {}
    for key in left.d.keys() | right.d.keys():
        out[key] = np.where(
            condition,
            left.d.get(key, 0.0),
            right.d.get(key, 0.0),
        )
    return Dual(np.where(condition, left.v, right.v), out)


def maximum(a: Operand, b: Operand) -> Dual:
    """Поэлементный максимум; на равенстве берётся ``b`` (как ``m_max`` LibXC)."""
    left, right = as_dual(a), as_dual(b)
    return where(left.v > right.v, left, right)


def minimum(a: Operand, b: Operand) -> Dual:
    """Поэлементный минимум; на равенстве берётся ``b`` (как ``m_min`` LibXC)."""
    left, right = as_dual(a), as_dual(b)
    return where(left.v > right.v, right, left)


def total_sum(terms: list[Dual]) -> Dual:
    """Сумма списка слагаемых."""
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result
