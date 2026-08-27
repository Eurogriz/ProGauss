"""Физические константы и единицы.

Внутри расчётного ядра **все величины в атомной системе единиц** (Хартри).
Ангстремы, электрон-вольты, см⁻¹ и дебаи появляются только на границе с
пользователем, и это всегда явное преобразование — никогда «по умолчанию».
"""

from __future__ import annotations

import math

#: 1 Å в борах (CODATA).
ANGSTROM_TO_BOHR: float = 1.8897259885789

#: 1 бор в Å.
BOHR_TO_ANGSTROM: float = 1.0 / ANGSTROM_TO_BOHR

#: 1 Хартри в эВ (CODATA).
HARTREE_TO_EV: float = 27.211386245988

#: 1 Хартри в см⁻¹ (CODATA).
HARTREE_TO_CM1: float = 219474.6313632

#: 1 а.е. дипольного момента в дебаях.
AU_TO_DEBYE: float = 2.5417464739297717

#: π^{5/2} — множитель в выражении для двухэлектронных интегралов.
PI_5_2: float = math.pi**2.5


def angstrom_to_bohr(value: float) -> float:
    """Переводит ангстремы в боры."""
    return value * ANGSTROM_TO_BOHR


def bohr_to_angstrom(value: float) -> float:
    """Переводит боры в ангстремы."""
    return value * BOHR_TO_ANGSTROM


def hartree_to_ev(value: float) -> float:
    """Переводит хартри в электрон-вольты."""
    return value * HARTREE_TO_EV


def hartree_to_cm1(value: float) -> float:
    """Переводит хартри в обратные сантиметры."""
    return value * HARTREE_TO_CM1


def au_to_debye(value: float) -> float:
    """Переводит атомные единицы дипольного момента в дебаи."""
    return value * AU_TO_DEBYE
