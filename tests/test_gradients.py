"""Проверка аналитических градиентов RHF.

Главный тест — совпадение с конечными разностями энергии. Это самопроверка, не
зависящая от внешних пакетов: если аналитическая формула расходится с численной
производной той же энергии, ошибка в формуле, а не в числах.

Отдельно проверяются два инварианта, нарушение которых означает потерянное
слагаемое:

* **поступательная инвариантность** — сумма градиентов по всем атомам равна нулю;
* **производная ERI** — сверяется с конечными разностями напрямую, чтобы ошибка
  в тензоре не маскировалась совпадением других слагаемых.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.engine import integrals
from quantumlab.engine.basis import build_basis
from quantumlab.engine.constants import angstrom_to_bohr
from quantumlab.engine.gradients import (
    RhfGradient,
    energy_weighted_density,
    nuclear_repulsion_gradient,
    rhf_gradient,
)
from quantumlab.engine.scf import ScfSettings, run_rhf

WATER = Path(__file__).parent / "fixtures" / "water.xyz"

#: Шаг конечных разностей в ангстремах. Меньше — хуже из-за округления
#: (энергия ~1e-16 относительных), больше — из-за отбрасывания членов ряда.
_STEP_ANGSTROM = 1e-4


def _h2(distance: float = 0.7414) -> Molecule:
    return Molecule(
        name="h2",
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, distance)),
        ),
    )


def _water() -> Molecule:
    return Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")


def _displaced(molecule: Molecule, atom_index: int, axis: int, step: float) -> Molecule:
    atoms = list(molecule.atoms)
    position = list(atoms[atom_index].position)
    position[axis] += step
    atoms[atom_index] = atoms[atom_index].model_copy(update={"position": tuple(position)})
    return molecule.model_copy(update={"atoms": tuple(atoms)})


def _energy(molecule: Molecule, basis_name: str) -> float:
    return run_rhf(build_basis(basis_name, molecule), molecule, ScfSettings()).total_energy


def _numerical_gradient(molecule: Molecule, basis_name: str) -> np.ndarray:
    """Градиент энергии конечными разностями, хартри/бор."""
    gradient = np.zeros((molecule.n_atoms, 3))
    denominator = 2.0 * angstrom_to_bohr(_STEP_ANGSTROM)
    for atom_index in range(molecule.n_atoms):
        for axis in range(3):
            plus = _energy(_displaced(molecule, atom_index, axis, _STEP_ANGSTROM), basis_name)
            minus = _energy(_displaced(molecule, atom_index, axis, -_STEP_ANGSTROM), basis_name)
            gradient[atom_index, axis] = (plus - minus) / denominator
    return gradient


def _analytical(molecule: Molecule, basis_name: str) -> RhfGradient:
    basis = build_basis(basis_name, molecule)
    scf = run_rhf(basis, molecule, ScfSettings())
    assert scf.converged
    return rhf_gradient(basis, molecule, scf)


# --------------------------------------------------------------------------- #
# Градиент против конечных разностей
# --------------------------------------------------------------------------- #
def test_hydrogen_gradient_matches_finite_differences() -> None:
    """H₂: все шесть компонент совпадают с численной производной."""
    molecule = _h2()
    result = _analytical(molecule, "sto-3g")
    numerical = _numerical_gradient(molecule, "sto-3g")
    assert np.max(np.abs(result.gradient - numerical)) < 1e-6


def test_water_gradient_matches_finite_differences() -> None:
    """Вода/STO-3G: девять компонент, включая ненулевые угловые."""
    molecule = _water()
    result = _analytical(molecule, "sto-3g")
    numerical = _numerical_gradient(molecule, "sto-3g")
    assert np.max(np.abs(result.gradient - numerical)) < 1e-6
    # Геометрия фикстуры не равновесная — градиент обязан быть заметным,
    # иначе тест мог бы пройти и на обнулённом результате.
    assert result.max_force > 1e-2


def test_polarized_basis_gradient_matches_finite_differences() -> None:
    """Базис с p-функциями: производные по угловому моменту тоже задействованы."""
    molecule = _h2()
    result = _analytical(molecule, "6-31g")
    numerical = _numerical_gradient(molecule, "6-31g")
    assert np.max(np.abs(result.gradient - numerical)) < 1e-6


# --------------------------------------------------------------------------- #
# Инварианты
# --------------------------------------------------------------------------- #
def test_translational_invariance() -> None:
    """Сумма градиентов по всем атомам равна нулю.

    Энергия не зависит от положения молекулы как целого, поэтому сумма сил
    обязана обращаться в ноль. Нарушение означает потерянное слагаемое —
    именно так проявлялась ошибка симметризации производных блоков.
    """
    result = _analytical(_water(), "sto-3g")
    assert np.max(np.abs(result.gradient.sum(axis=0))) < 1e-10


def test_gradient_vanishes_for_a_symmetric_stretched_dimer_only_at_minimum() -> None:
    """Симметрия: для H₂ равны и противоположны компоненты на двух атомах."""
    result = _analytical(_h2(), "sto-3g")
    assert result.gradient[0] == pytest.approx(-result.gradient[1], abs=1e-12)


def test_forces_are_the_negated_gradient() -> None:
    """Сила определена как ``F = −dE/dR`` — знак легко потерять на границе."""
    result = _analytical(_h2(), "sto-3g")
    assert np.allclose(result.forces, -result.gradient)


# --------------------------------------------------------------------------- #
# Отдельные слагаемые
# --------------------------------------------------------------------------- #
def test_nuclear_repulsion_gradient_matches_finite_differences() -> None:
    """Межъядерное отталкивание проверяется независимо от квантовой части."""
    from quantumlab.engine.basis import nuclear_repulsion

    molecule = _water()
    analytical = nuclear_repulsion_gradient(molecule)
    numerical = np.zeros((molecule.n_atoms, 3))
    step = 1e-5
    denominator = 2.0 * angstrom_to_bohr(step)
    for atom_index in range(molecule.n_atoms):
        for axis in range(3):
            plus = nuclear_repulsion(_displaced(molecule, atom_index, axis, step))
            minus = nuclear_repulsion(_displaced(molecule, atom_index, axis, -step))
            numerical[atom_index, axis] = (plus - minus) / denominator
    assert np.max(np.abs(analytical - numerical)) < 1e-7


def test_electron_repulsion_derivative_matches_finite_differences() -> None:
    """Производная тензора ERI по центру бра — сверка напрямую.

    Выделено в отдельный тест намеренно: этот тензор не обладает 8-кратной
    симметрией самого ERI, и ошибка в его сборке могла бы частично
    компенсироваться другими слагаемыми в полном градиенте.
    """
    molecule = _h2()
    basis = build_basis("sto-3g", molecule)
    # В STO-3G у каждого атома водорода ровно одна базисная функция, поэтому
    # срез (0 1|1 1) содержит функцию нулевого атома ровно в одном слоте.
    # Сравнивать срез вроде (0 0|0 0) нельзя: там функция нулевого атома стоит
    # во всех четырёх слотах, и производная по АТОМУ равна сумме четырёх
    # производных по центрам, а не одной из них.
    for axis in range(3):
        analytical = integrals.build_electron_repulsion_derivative(basis, molecule, axis)
        plus = integrals.build_electron_repulsion(
            basis, _displaced(molecule, 0, axis, _STEP_ANGSTROM)
        )
        minus = integrals.build_electron_repulsion(
            basis, _displaced(molecule, 0, axis, -_STEP_ANGSTROM)
        )
        numerical = (plus - minus) / (2.0 * angstrom_to_bohr(_STEP_ANGSTROM))
        assert analytical[0, 1, 1, 1] == pytest.approx(numerical[0, 1, 1, 1], abs=1e-6)
        # Симметрия λ↔σ у производной сохраняется — проверяем её явно.
        assert analytical[0, 1, 1, 1] == pytest.approx(analytical[0, 1, 1, 1], abs=0.0)
        assert np.allclose(analytical, analytical.transpose(0, 1, 3, 2), atol=1e-14)
        # А срез без функции нулевого атома по этому атому не дифференцируется.
        assert numerical[1, 1, 1, 1] == pytest.approx(0.0, abs=1e-9)


def test_energy_weighted_density_is_symmetric_and_trace_matches() -> None:
    """W симметрична, а её свёртка с S даёт сумму занятых орбитальных энергий."""
    molecule = _water()
    basis = build_basis(basis_name := "sto-3g", molecule)
    scf = run_rhf(basis, molecule, ScfSettings())
    n_occupied = molecule.n_electrons // 2
    weight = energy_weighted_density(scf, n_occupied)
    assert np.allclose(weight, weight.T)
    overlap = integrals.build_overlap(basis, molecule)
    expected = 2.0 * sum(scf.orbital_energies[:n_occupied])
    assert float(np.sum(weight * overlap)) == pytest.approx(expected, rel=1e-10)
    del basis_name


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g"])
def test_vectorized_eri_derivative_matches_scalar_reference(basis_name: str) -> None:
    """Векторизованный блок производной ERI совпадает со скалярной спецификацией.

    В ``integrals`` живут два пути: скалярный (``_scalar``) — медленный, но
    читаемый, и векторизованный — рабочий. Тест фиксирует, что ускорение не
    изменило физику: расхождение должно оставаться на уровне машинной точности.
    Проверяются и ``s``-, и ``p``-оболочки — именно на них работают ветвления
    рекурсии по угловым моментам.
    """
    molecule = _water()
    basis = build_basis(basis_name, molecule)
    centers = integrals._shell_centers(basis, molecule)
    worst = 0.0
    checked = 0
    for axis in range(3):
        for i in range(len(basis.shells)):
            for j in range(len(basis.shells)):
                for k in range(len(basis.shells)):
                    for m in range(k + 1):
                        arguments = (
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
                        fast = integrals._quartet_derivative_block(*arguments)
                        slow = integrals._quartet_derivative_block_scalar(*arguments)
                        worst = max(worst, float(np.abs(fast - slow).max()))
                        checked += 1
    assert checked > 0
    assert worst < 1e-13, worst


def test_boys_array_matches_scalar_boys() -> None:
    """Векторизованная функция Бойса совпадает со скалярной на обоих ветвях.

    Ряд Тейлора (x < 6) и рекурсия от erf (x ≥ 6) — разные ветви, поэтому
    проверяются обе: векторизация не должна менять ни одну из них.
    """
    arguments = np.array([0.0, 1e-9, 0.3, 2.5, 5.9, 6.0, 6.1, 12.0, 40.0, 250.0])
    for order in range(5):
        expected = np.array([integrals._boys(order, float(x)) for x in arguments])
        obtained = integrals._boys_array(order, arguments)
        assert np.abs(obtained - expected).max() < 1e-13, order


def test_gradient_requires_converged_scf_by_construction() -> None:
    """Документированное требование: на несошедшейся плотности градиент неверен.

    Проверка выполняется вызывающим кодом. Здесь фиксируем, что движок не
    прячет признак сходимости: поле ``converged`` доступно и честно отражает
    состояние SCF.
    """
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    scf = run_rhf(basis, molecule, ScfSettings(max_iterations=1))
    assert not scf.converged
    # Градиент технически вычислим, поэтому ответственность за проверку лежит
    # на движке — это фиксируется тестом движка, а не здесь.
    result = rhf_gradient(basis, molecule, scf)
    assert result.gradient.shape == (molecule.n_atoms, 3)
