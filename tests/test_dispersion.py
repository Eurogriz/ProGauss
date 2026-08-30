"""Дисперсионная поправка DFT-D3: свойства, честность, аналитический градиент.

Научные сверки с независимой сборкой s-dftd3 вынесены в
``test_crosscheck_dftd3.py`` (требует PySCF и помечены ``scientific``).
Здесь — то, что проверяется без внешних пакетов:

* табличные данные (сгенерированы, сверены с исходниками генератором);
* честность: недоступный метод отклоняется явной ошибкой, а не приближается;
* аналитический градиент: сравнение с конечными разностями (1e-10) — это
  проверка математической сборки, а не физики;
* трансляционная инвариантность: сумма сил по всем атомам равна нулю.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.domain.spec import DispersionCorrection
from quantumlab.engine import dispersion as d3
from quantumlab.engine import dispersion_data as dd
from quantumlab.engine.constants import BOHR_TO_ANGSTROM

#: Шаг конечных разностей по координате, bohr: достаточно мал для 5-точечного
#: шаблона (ошибка O(h⁴) ~ 1e-12), достаточно велик для шума округления.
_FD_STEP_BOHR: float = 1e-5
#: Допуск совпадения аналитического и конечноразностного градиента. Это
#: проверка внутренней согласованности (градиент — производная той же
#: энергии, что и в оптимизаторе), а не проверка физики: точная сверка с
#: независимой реализацией живёт в test_crosscheck_dftd3.py. FD-шаблон на
#: крутом 1/r⁶-потенциале теряет ~1e-10..1e-9, поэтому допуск выше его
#: предельной ошибки, но на четыре порядка ниже самой величины градиента.
_FD_TOL: float = 1e-8
#: Абсолютный пол, э/bohr: шум округления пяти близких энергий, делённый на
#: шаг, задаёт нижнюю границу, ниже которой «относительная» погрешность на
#: почти нулевом градиенте бессмысленна.
_FD_ABS_TOL: float = 1e-12


def _assert_gradient_close(analytic: np.ndarray, fd: np.ndarray) -> None:
    diff = float(np.max(np.abs(analytic - fd)))
    scale = max(1e-30, float(np.max(np.abs(analytic))))
    assert diff <= _FD_ABS_TOL + _FD_TOL * scale, f"|∇анал − ∇FD| = {diff:.2e}, масштаб {scale:.2e}"


#: Молекула «лента» из всех 12 элементов области применения: проверка, что
#: таблица C6 и ковалентных радиусов покрывает весь домен без исключений.
_ALL_ELEMENTS = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I")


def _strip() -> Molecule:
    """Атомы в ряд с шагом 1.5 Å — все пары в области действия модели."""
    atoms = tuple(Atom(symbol=s, position=(1.5 * i, 0.0, 0.0)) for i, s in enumerate(_ALL_ELEMENTS))
    return Molecule(atoms=atoms)


def _water() -> Molecule:
    return Molecule(
        atoms=(
            Atom(symbol="O", position=(0.0, 0.0, 0.1173)),
            Atom(symbol="H", position=(0.0, 0.7572, -0.4692)),
            Atom(symbol="H", position=(0.0, -0.7572, -0.4692)),
        )
    )


def _h2(distance_angstrom: float) -> Molecule:
    return Molecule(
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(distance_angstrom, 0.0, 0.0)),
        )
    )


# --------------------------------------------------------------------------- #
# Табличные данные
# --------------------------------------------------------------------------- #
def test_element_table_covers_domain() -> None:
    assert tuple(dd.D3_ELEMENTS) == tuple(element.z for element in (_strip().atoms))
    for z in dd.D3_ELEMENTS:
        el = dd.element_data(z)
        assert el.nref >= 1
        assert len(el.refcn) == el.nref
        assert el.r4eff == pytest.approx(math.sqrt(0.5 * el.r4raw * math.sqrt(z)), rel=1e-15)
        # Ковалентный радиус s-dftd3 = Pyykko 2009 × 4/3 (mctc-lib, covalent_rad_d3).
        assert el.rcov_bohr == pytest.approx(4.0 / 3.0 * el.rcov_2009_angstrom * dd.AATOA_D3)


def test_c6_table_known_values() -> None:
    # H–H: значения систем отсчёта s-dftd3 v1.2.1 (таблица C6, пара H/H).
    assert dd.C6_PAIRS[(1, 1)] == pytest.approx((3.0267, 4.7379, 4.7379, 7.5916), rel=1e-12)
    # Симметрия парного индекса: c6(i, j) = c6(j, i) как величина.
    assert dd.vdw_pair(1, 8) == pytest.approx(dd.vdw_pair(8, 1))


def test_damping_parameter_tables() -> None:
    # s6 = 1 у всех обученных параметров (s-dftd3, param.f90).
    for params in dd.BJ_PARAMS.values():
        assert params[0] == pytest.approx(1.0)
    for zero_params in dd.ZERO_PARAMS.values():
        assert zero_params[0] == pytest.approx(1.0)
        assert zero_params[4] == 14  # альфа: степень затухания 6-го члена
    # Fallback-функционалы движка покрыты: hf, pbe, pbe0, blyp, b3lyp.
    assert set(dd.BJ_PARAMS) == {"hf", "pbe", "pbe0", "blyp", "b3lyp"}
    assert set(dd.ZERO_PARAMS) == {"hf", "pbe", "pbe0", "blyp", "b3lyp"}


# --------------------------------------------------------------------------- #
# Честность: недоступный метод — явная ошибка, а не приближение (§54 ТЗ)
# --------------------------------------------------------------------------- #
def test_lda_has_no_d3_params_and_is_rejected() -> None:
    with pytest.raises(ValueError, match="не обучен для функционала"):
        d3.dftd3_contribution(_water(), DispersionCorrection.D3_BJ, "svwn")
    with pytest.raises(ValueError, match="не обучен для функционала"):
        d3.dftd3_contribution(_water(), DispersionCorrection.D3_ZERO, "lda")


def test_unknown_correction_is_rejected() -> None:
    with pytest.raises(ValueError):
        d3.dftd3_contribution(_water(), DispersionCorrection.D4, "pbe")


def test_out_of_domain_element_guard() -> None:
    # Страховка на случай расширения домена элементов: атом с Z вне области
    # применения должен отклоняться списком символов, а не «тихим» расчётом.
    fake_atom = SimpleNamespace(z=4, symbol="Be")
    fake_molecule = SimpleNamespace(atoms=(fake_atom,))
    with pytest.raises(ValueError, match="вне области применения"):
        # Двойник Molecule: функция читает только `.atoms[i].z/.symbol`.
        d3._check_elements(fake_molecule)  # type: ignore[arg-type]


def test_functionals_listing_matches_tables() -> None:
    assert d3.dftd3_functionals(DispersionCorrection.D3_BJ) == tuple(dd.BJ_PARAMS)
    assert d3.dftd3_functionals(DispersionCorrection.D3_ZERO) == tuple(dd.ZERO_PARAMS)


# --------------------------------------------------------------------------- #
# Свойства вклады
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("correction", [DispersionCorrection.D3_BJ, DispersionCorrection.D3_ZERO])
@pytest.mark.parametrize("functional", [None, "pbe", "pbe0", "blyp", "b3lyp"])
def test_contribution_is_negative_and_finite(
    correction: DispersionCorrection, functional: str | None
) -> None:
    for molecule in (_water(), _strip()):
        contribution = d3.dftd3_contribution(molecule, correction, functional)
        assert contribution.energy_hartree < 0.0
        assert math.isfinite(contribution.energy_hartree)
        assert contribution.gradient.shape == (molecule.n_atoms, 3)
        assert np.all(np.isfinite(contribution.gradient))


def test_far_away_atoms_give_no_contribution() -> None:
    # Дальше отсечения 60 bohr пара не входит в сумму: вклад — машинный ноль.
    molecule = Molecule(
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(40.0, 0.0, 0.0)),  # ~75 bohr
        )
    )
    contribution = d3.dftd3_contribution(molecule, DispersionCorrection.D3_BJ, None)
    assert contribution.energy_hartree == 0.0
    assert np.max(np.abs(contribution.gradient)) == 0.0


def test_bj_damping_is_finite_at_small_distance() -> None:
    # BJ — рациональная функция: в r → 0 энергия конечна (не расходится).
    contribution = d3.dftd3_contribution(_h2(0.05), DispersionCorrection.D3_BJ, None)
    assert math.isfinite(contribution.energy_hartree)
    assert np.all(np.isfinite(contribution.gradient))


def test_translational_invariance() -> None:
    # Каждая пара даёт +F и −F, поэтому сумма сил строго нулевая:
    # ненулевой остаток — артефакт сборки, а не физика.
    for correction in (DispersionCorrection.D3_BJ, DispersionCorrection.D3_ZERO):
        contribution = d3.dftd3_contribution(_strip(), correction, "pbe")
        assert np.max(np.abs(contribution.gradient.sum(axis=0))) < 1e-12


# --------------------------------------------------------------------------- #
# Аналитический градиент против конечных разностей
# --------------------------------------------------------------------------- #
def _d3_energy(
    molecule: Molecule, correction: DispersionCorrection, functional: str | None
) -> float:
    return d3.dftd3_contribution(molecule, correction, functional).energy_hartree


def _fd_gradient(
    molecule: Molecule, correction: DispersionCorrection, functional: str | None
) -> np.ndarray:
    """Градиент центральными разностями, результат в э/bohr.

    Координаты молекулы в Å, поэтому шаг по координате — в Å; но результат
    обязан быть производной по бохру (как аналитический градиент): деление
    идёт на шаг, выраженный в бохрах.
    """
    n = molecule.n_atoms
    gradient = np.zeros((n, 3))
    h = _FD_STEP_BOHR * BOHR_TO_ANGSTROM
    for i in range(n):
        for axis in range(3):
            # 5-точечный центральный шаблон O(h⁴): на крутом 1/r⁶-потенциале
            # даёт точность ~1e-12 против ~1e-9 у двухточечного.
            e_m2 = _d3_energy(_shift(molecule, i, axis, -2.0 * h), correction, functional)
            e_m1 = _d3_energy(_shift(molecule, i, axis, -h), correction, functional)
            e_p1 = _d3_energy(_shift(molecule, i, axis, +h), correction, functional)
            e_p2 = _d3_energy(_shift(molecule, i, axis, +2.0 * h), correction, functional)
            gradient[i, axis] = (-e_p2 + 8.0 * e_p1 - 8.0 * e_m1 + e_m2) / (12.0 * _FD_STEP_BOHR)
    return gradient


def _shift(molecule: Molecule, atom: int, axis: int, delta: float) -> Molecule:
    atoms = list(molecule.atoms)
    position = list(atoms[atom].position)
    position[axis] += delta
    atoms[atom] = Atom(symbol=atoms[atom].symbol, position=(position[0], position[1], position[2]))
    return Molecule(atoms=tuple(atoms))


@pytest.mark.parametrize("correction", [DispersionCorrection.D3_BJ, DispersionCorrection.D3_ZERO])
@pytest.mark.parametrize("functional", [None, "pbe"])
def test_analytic_gradient_matches_finite_differences(
    correction: DispersionCorrection, functional: str | None
) -> None:
    for molecule in (_water(), _h2(0.7), _strip()):
        contribution = d3.dftd3_contribution(molecule, correction, functional)
        fd = _fd_gradient(molecule, correction, functional)
        _assert_gradient_close(contribution.gradient, fd)
