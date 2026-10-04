"""Дисперсионная поправка DFT-D2 с затуханием Чая—Хед-Гордона (часть ωB97X-D).

ωB97X-D — это не «ωB97X плюс любая дисперсия»: поправка входит в определение
функционала (Chai, Head-Gordon, PCCP 10, 6615 (2008))::

    E_disp = −Σ_{i<j} C6_ij / R_ij⁶ · 1 / (1 + a (R_ij / R_r)⁻¹²),
    C6_ij = √(C6_i C6_j),  R_r = R0_i + R0_j,  a = 6.0,

с атомными ``C6`` и ``R0`` второй параметризации Гримме (J. Comput. Chem. 27,
1787 (2006)), ``s6 = 1``. Отличие от D2 Гримме — только затухание: у Гримме
``1/(1 + exp(−d(R/R_r − 1)))``, у CHG — степенное, и оно не обращается в ноль
на малых ``R``, а ограничивает вклад пропорционально ``R⁶/a``.

Ограничение честности: независимого программного оракула для этой поправки в
окружении нет (PySCF не реализует D2/CHG, а s-dftd3 и dftd4 — это D3/D4), поэтому
формула проверяется аналитически (тест считает сумму пар вручную), а градиент —
конечными разностями. Таблица параметров сверена с таблицей 6.3 руководства
Turbomole (параметры Гримме 2006).
"""

from __future__ import annotations

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.constants import ANGSTROM_TO_BOHR
from quantumlab.engine.dispersion import DispersionContribution

#: Параметр затухания CHG для ωB97X-D.
CHG_DAMPING: float = 6.0

#: ``C6`` (Дж·нм⁶/моль) и ``R0`` (Å) по атомному номеру: Гримме 2006, элементы H–Xe.
_PARAMETERS: dict[int, tuple[float, float]] = {
    1: (0.14, 1.001),
    2: (0.08, 1.012),
    3: (1.61, 0.825),
    4: (1.61, 1.408),
    5: (3.13, 1.485),
    6: (1.75, 1.452),
    7: (1.23, 1.397),
    8: (0.70, 1.342),
    9: (0.75, 1.287),
    10: (0.63, 1.243),
    11: (5.71, 1.144),
    12: (5.71, 1.364),
    13: (10.79, 1.639),
    14: (9.23, 1.716),
    15: (7.84, 1.705),
    16: (5.57, 1.683),
    17: (5.07, 1.639),
    18: (4.61, 1.595),
    19: (10.80, 1.485),
    20: (10.80, 1.474),
    31: (16.99, 1.650),
    32: (17.10, 1.727),
    33: (16.37, 1.760),
    34: (12.64, 1.771),
    35: (12.47, 1.749),
    36: (12.01, 1.727),
    37: (24.67, 1.628),
    38: (24.67, 1.606),
    49: (37.32, 1.672),
    50: (38.71, 1.804),
    51: (38.44, 1.881),
    52: (31.74, 1.892),
    53: (31.50, 1.892),
    54: (29.99, 1.881),
}
_PARAMETERS.update(dict.fromkeys(range(21, 31), (10.8, 1.562)))
_PARAMETERS.update(dict.fromkeys(range(39, 49), (24.67, 1.639)))

#: 1 Дж/моль в хартри: ``1 / (N_A · E_h)`` (CODATA 2018).
_JOULE_PER_MOL_TO_HARTREE: float = 1.0 / (6.02214076e23 * 4.3597447222071e-18)
#: 1 нм в борах.
_NM_TO_BOHR: float = 10.0 * ANGSTROM_TO_BOHR
#: ``C6``: Дж·нм⁶/моль → хартри·бор⁶.
_C6_UNIT: float = _JOULE_PER_MOL_TO_HARTREE * _NM_TO_BOHR**6


def chg_elements() -> tuple[int, ...]:
    """Атомные номера, для которых есть параметры D2."""
    return tuple(sorted(_PARAMETERS))


def dftd2_chg_contribution(molecule: Molecule) -> DispersionContribution:
    """Энергия и аналитический градиент поправки D2-CHG (хартри, хартри/бор)."""
    missing = {atom.symbol for atom in molecule.atoms if atom.z not in _PARAMETERS}
    if missing:
        msg = (
            f"Для элемента(ов) {', '.join(sorted(missing))} нет параметров D2 (Гримме 2006): "
            "ωB97X-D без дисперсионной поправки — другой метод, поэтому запрос отклонён."
        )
        raise ValueError(msg)
    c6 = np.array([_PARAMETERS[atom.z][0] * _C6_UNIT for atom in molecule.atoms])
    radius = np.array([_PARAMETERS[atom.z][1] * ANGSTROM_TO_BOHR for atom in molecule.atoms])
    positions = np.array([atom.position for atom in molecule.atoms], dtype=float) * ANGSTROM_TO_BOHR
    n_atoms = len(molecule.atoms)
    energy = 0.0
    gradient = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            delta = positions[i] - positions[j]
            distance = float(np.linalg.norm(delta))
            if distance < 1e-6:
                continue
            c6_pair = float(np.sqrt(c6[i] * c6[j]))
            ratio = ((radius[i] + radius[j]) / distance) ** 12
            damping = 1.0 / (1.0 + CHG_DAMPING * ratio)
            energy -= c6_pair * damping / distance**6
            # dE/dR = (C6/R⁷)(6 f − 12 a u f²), u = (R_r/R)¹².
            slope = (
                c6_pair / distance**7 * (6.0 * damping - 12.0 * CHG_DAMPING * ratio * damping**2)
            )
            pair_force = slope * delta / distance
            gradient[i] += pair_force
            gradient[j] -= pair_force
    return DispersionContribution(
        model="d2chg", functional="wb97x-d", energy_hartree=energy, gradient=gradient
    )
