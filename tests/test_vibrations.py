"""Проверка колебательного анализа.

Главная сверка — с аналитическим гессианом PySCF: наш гессиан численный, и
без независимого оракула ошибка в шаге разностей выглядела бы как физика.
Отдельно проверяются инварианты, нарушение которых означает потерянную
проекцию или неверные единицы:

* число спроецированных мод — 6 для нелинейной молекулы и 5 для линейной;
* перенос всей молекулы не меняет частот;
* мнимые частоты не отбрасываются, а возвращаются отрицательными.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quantumlab.domain.molecule import ELEMENTS, Atom, Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.constants import AMU_TO_ELECTRON_MASS
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.gradients import rhf_gradient
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import ScfSettings, run_rhf
from quantumlab.engine.vibrations import (
    numerical_hessian,
    rigid_body_modes,
    vibrational_analysis,
)

WATER = Path(__file__).parent / "fixtures" / "water.xyz"
TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)

pytestmark = pytest.mark.scientific


def _pyscf_thermo() -> Any:
    """Модуль термохимии PySCF.

    Подключается через ``importlib``: статический импорт заставил бы mypy искать
    у PySCF заглушки типов, которых нет.
    """
    return importlib.import_module("pyscf.hessian.thermo")


def _water() -> Molecule:
    return Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")


def _rhf_gradient(molecule: Molecule) -> np.ndarray:
    basis = build_basis("sto-3g", molecule)
    return rhf_gradient(basis, molecule, run_rhf(basis, molecule, TIGHT)).gradient


def _h2(distance_angstrom: float = 0.74) -> Molecule:
    return Molecule(
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, distance_angstrom)),
        ),
        name="h2",
    )


def test_atomic_masses_match_independent_table() -> None:
    """Массы элементов сверяются с независимой таблицей, а не с памятью.

    Ошибка в массе сдвинула бы все частоты как корень из отношения масс —
    правдоподобно на вид и неверно.
    """
    pyscf = pytest.importorskip("pyscf")
    masses = pyscf.data.elements.MASSES
    for element in ELEMENTS:
        assert element.mass_amu == pytest.approx(masses[element.z], rel=1e-9), element.symbol


def test_rigid_body_modes_count_depends_on_geometry() -> None:
    """У нелинейной молекулы 6 жестких мод, у линейной — 5.

    Вращение вокруг собственной оси линейной молекулы не меняет ничего, поэтому
    соответствующий вектор зануляется и отсеивается при ортогонализации.
    """
    assert rigid_body_modes(_water()).shape[0] == 6
    assert rigid_body_modes(_h2()).shape[0] == 5


def test_rigid_body_modes_are_orthonormal() -> None:
    """Базис обязан быть ортонормированным: иначе проектор не проектор."""
    basis = rigid_body_modes(_water())
    product = basis @ basis.T
    assert np.max(np.abs(product - np.eye(basis.shape[0]))) < 1e-12


def test_frequencies_are_invariant_to_translation() -> None:
    """Перенос молекулы целиком не меняет частот."""
    original = vibrational_analysis(
        numerical_hessian(_water(), _rhf_gradient), _water()
    ).frequencies_cm1

    shifted = Molecule(
        atoms=tuple(
            Atom(
                symbol=atom.symbol,
                position=(
                    atom.position[0] + 0.7,
                    atom.position[1] + 0.7,
                    atom.position[2] + 0.7,
                ),
            )
            for atom in _water().atoms
        ),
        name="water-shifted",
    )
    moved = vibrational_analysis(numerical_hessian(shifted, _rhf_gradient), shifted)
    # Не ноль: гессиан численный, и сдвиг меняет картину округлений. Порог выбран
    # так, чтобы сломанная проекция (она дала бы расхождение на порядки) всё
    # равно проваливалась.
    assert np.max(np.abs(np.array(moved.frequencies_cm1) - np.array(original))) < 0.01


def test_hessian_matches_pyscf_analytic_hessian() -> None:
    """Численный гессиан против аналитического гессиана PySCF.

    PySCF возвращает тензор в раскладке ``(A, B, a, b)``, а не ``(A, a, B, b)``:
    при неправильной перестановке матрица становится несимметричной, и это
    единственный способ заметить ошибку на молекуле, где число атомов равно 3.
    """
    pyscf = pytest.importorskip("pyscf")
    molecule = _water()
    ours = numerical_hessian(molecule, _rhf_gradient)

    atom_string = "; ".join(
        f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}"
        for atom in molecule.atoms
    )
    reference = pyscf.scf.RHF(
        pyscf.gto.M(atom=atom_string, basis="sto-3g", cart=True, verbose=0)
    ).run(conv_tol=1e-12)
    theirs = np.asarray(reference.Hessian().kernel()).transpose(0, 2, 1, 3).reshape(9, 9)

    assert np.max(np.abs(theirs - theirs.T)) < 1e-10, "раскладка PySCF изменилась"
    assert float(np.max(np.abs(ours - theirs))) < 1e-5


def test_vibrational_analysis_reproduces_pyscf_frequencies() -> None:
    """Наш колебательный анализ на гессиане PySCF обязан дать ровно его частоты.

    Проверка отделяет масс-взвешивание, проекцию и перевод единиц от качества
    самого гессиана: вход один и тот же, значит любое расхождение — в анализе.
    """
    pyscf = pytest.importorskip("pyscf")
    molecule = _water()
    atom_string = "; ".join(
        f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}"
        for atom in molecule.atoms
    )
    mol = pyscf.gto.M(atom=atom_string, basis="sto-3g", cart=True, verbose=0)
    reference = pyscf.scf.RHF(mol).run(conv_tol=1e-12)
    raw = np.asarray(reference.Hessian().kernel())

    expected = np.asarray(_pyscf_thermo().harmonic_analysis(mol, raw)["freq_wavenumber"])
    ours = vibrational_analysis(raw.transpose(0, 2, 1, 3).reshape(9, 9), molecule)
    relative = float(np.max(np.abs(np.array(ours.frequencies_cm1) - expected) / expected))
    # Расхождение 1.1e-09 относительных и одинаковое для всех трёх мод — значит
    # дело в глобальной константе, а не в проекции или масс-взвешивании. PySCF
    # берёт HARTREE2WAVENUMBER = 219474.63111558527, мы — CODATA 2018
    # (219474.6313632); плюс разница версий AMU2AU. Физически несущественно,
    # но допуск обязан это учитывать, а не притворяться, что совпадение точное.
    assert relative < 1e-7, relative


def test_wavenumber_conversion_has_no_extra_two_pi() -> None:
    """Перевод в см⁻¹ — без лишнего деления на 2π.

    ``HARTREE_TO_CM1`` — это ``E_h/(hc)``, то есть ``h = 2πħ`` уже учтено, а при
    ``ħ = 1`` собственное значение масс-взвешенного гессиана численно равно
    энергии. Лишний ``2π`` сдвинул бы все частоты в 6.28 раза, и это легко
    спутать с «другим функционалом».
    """
    pyscf = pytest.importorskip("pyscf")
    molecule = _water()
    atom_string = "; ".join(
        f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}"
        for atom in molecule.atoms
    )
    mol = pyscf.gto.M(atom=atom_string, basis="sto-3g", cart=True, verbose=0)
    raw = np.asarray(pyscf.scf.RHF(mol).run(conv_tol=1e-12).Hessian().kernel())
    expected = np.asarray(_pyscf_thermo().harmonic_analysis(mol, raw)["freq_wavenumber"])

    ours = vibrational_analysis(raw.transpose(0, 2, 1, 3).reshape(9, 9), molecule)
    ratio = np.array(ours.frequencies_cm1) / expected
    assert np.max(np.abs(ratio - 1.0)) < 1e-7
    assert abs(float(ratio[0]) - 1 / (2 * math.pi)) > 0.1, "совпало с вариантом с 2π"


def test_negative_eigenvalue_is_reported_as_imaginary_frequency() -> None:
    """Мнимая частота возвращается отрицательной, а не отбрасывается.

    Проверка на перевёрнутом гессиане H₂: единственная колебательная мода
    обязана стать мнимой. Седловая точка, выданная за минимум, — ровно тот
    класс ошибок, который §54 ТЗ запрещает замалчивать.
    """
    molecule = _h2()
    hessian = numerical_hessian(molecule, _rhf_gradient)
    inverted = vibrational_analysis(-hessian, molecule)
    assert len(inverted.frequencies_cm1) == 1
    assert inverted.frequencies_cm1[0] < 0.0
    assert inverted.imaginary_frequencies == inverted.frequencies_cm1
    assert inverted.zero_point_energy_hartree == 0.0


def test_water_frequencies_from_own_hessian_match_pyscf() -> None:
    """Полный путь: наш численный гессиан → наш анализ → частоты PySCF."""
    pyscf = pytest.importorskip("pyscf")
    molecule = _water()
    ours = vibrational_analysis(numerical_hessian(molecule, _rhf_gradient), molecule)

    atom_string = "; ".join(
        f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}"
        for atom in molecule.atoms
    )
    mol = pyscf.gto.M(atom=atom_string, basis="sto-3g", cart=True, verbose=0)
    raw = np.asarray(pyscf.scf.RHF(mol).run(conv_tol=1e-12).Hessian().kernel())
    expected = np.asarray(_pyscf_thermo().harmonic_analysis(mol, raw)["freq_wavenumber"])

    deviation = float(np.max(np.abs(np.array(ours.frequencies_cm1) - expected)))
    # Разница — точность центральных разностей, а не ошибка формулы: сам
    # гессиан сходится с аналитическим до 1e-5 э/бор².
    assert deviation < 0.05, deviation
    assert ours.zero_point_energy_hartree > 0.0
    assert ours.imaginary_frequencies == ()


def test_frequencies_task_warns_off_stationary_point() -> None:
    """Частоты вне стационарной точки считаются, но сопровождаются предупреждением.

    Движок не подменяет запрос: пользователь попросил частоты, а не геометрию.
    Молча оптимизировать — значит вернуть не то, что просили.
    """
    result = ReferenceEngine().run(
        _request(_water(), Task.FREQUENCIES),
    )
    assert len(result.frequencies_cm1) == 3
    assert result.zero_point_energy_hartree is not None
    assert any(warning.key == "warning.frequencies_off_stationary" for warning in result.warnings)


def test_atomic_mass_unit_conversion_is_codata() -> None:
    """Перевод а.е.м. в массы электрона — из CODATA, а не подобран."""
    assert pytest.approx(1822.888486217, rel=1e-9) == AMU_TO_ELECTRON_MASS


def _request(molecule: Molecule, task: Task) -> EngineRequest:
    return EngineRequest(
        job_id="job-test",
        molecule=molecule,
        spec=CalculationSpec(
            task=task,
            method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
        ),
    )
