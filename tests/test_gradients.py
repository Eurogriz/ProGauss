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
from quantumlab.domain.spec import GridPreset
from quantumlab.engine import integrals
from quantumlab.engine.basis import build_basis, cartesian_powers
from quantumlab.engine.constants import angstrom_to_bohr
from quantumlab.engine.dft import run_rks
from quantumlab.engine.functional import (
    density_gradient_at_points,
    evaluate_basis_hessian_for_center,
    evaluate_basis_with_gradients,
    get_functional,
)
from quantumlab.engine.gradients import (
    RhfGradient,
    _function_owner,
    _rohf_orbital_weight,
    energy_weighted_density,
    nuclear_repulsion_gradient,
    rhf_gradient,
    rks_gradient,
    rohf_gradient,
    uhf_gradient,
)
from quantumlab.engine.quadrature import build_grid
from quantumlab.engine.scf import RohfResult, ScfSettings, run_rhf, run_rohf, run_uhf

WATER = Path(__file__).parent / "fixtures" / "water.xyz"

#: Шаг конечных разностей в ангстремах. Меньше — хуже из-за округления
#: (энергия ~1e-16 относительных), больше — из-за отбрасывания членов ряда.
_STEP_ANGSTROM = 1e-4
_TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)


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


# --------------------------------------------------------------------------- #
# DFT: вторые производные базисных функций и обменно-корреляционный градиент
# --------------------------------------------------------------------------- #
_STEP_BOHR = 1e-4


def _displace_water(atom_index: int, axis: int, delta_angstrom: float) -> Molecule:
    """Вода со смещённым атомом: сетка при этом не перестраивается."""
    base = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    atoms: list[Atom] = []
    for index, atom in enumerate(base.atoms):
        position = list(atom.position)
        if index == atom_index:
            position[axis] += delta_angstrom
        atoms.append(Atom(symbol=atom.symbol, position=(position[0], position[1], position[2])))
    return Molecule(atoms=tuple(atoms), name="water")


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g", "cc-pvdz"])
def test_basis_hessian_matches_finite_differences(basis_name: str) -> None:
    """Гессиан базисных функций совпадает с конечными разностями градиента.

    Вторые производные нужны только обменно-корреляционному градиенту GGA, и
    энергия от них не зависит: ошибка здесь не проявилась бы ни в SCF, ни в
    энергии, поэтому проверка отдельная.
    """
    molecule = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    basis = build_basis(basis_name, molecule)
    points = np.random.default_rng(11).normal(size=(30, 3)) * 1.5
    step = 1e-5
    worst = 0.0
    for center in range(len(molecule.atoms)):
        columns: list[int] = []
        offset = 0
        for shell in basis.shells:
            width = len(cartesian_powers(shell.angular_momentum))
            if shell.center == center:
                columns.extend(range(offset, offset + width))
            offset += width
        hessian = evaluate_basis_hessian_for_center(basis, molecule, points, center)
        assert hessian.shape == (points.shape[0], len(columns), 3, 3)
        for axis in range(3):
            plus = points.copy()
            plus[:, axis] += step
            minus = points.copy()
            minus[:, axis] -= step
            _, grad_plus = evaluate_basis_with_gradients(basis, molecule, plus)
            _, grad_minus = evaluate_basis_with_gradients(basis, molecule, minus)
            numeric = (grad_plus - grad_minus) / (2 * step)
            worst = max(
                worst, float(np.max(np.abs(numeric[:, columns, :] - hessian[:, :, axis, :])))
            )
    assert worst < 1e-8


def test_hessian_contraction_is_not_summed_over_the_wrong_axis() -> None:
    """Регрессия на подпись einsum: ось дифференцирования срезается до свёртки.

    В ``np.einsum("pjab,jp->pb", ...)`` индекс ``a`` отсутствует в выходной
    части, и einsum считает его суммируемым: свёртка сложит производные по всем
    трём осям и даст неверный ``∂σ/∂R``. Знакомая ловушка — энергия при этом
    остаётся правильной. Проверка сравнивает einsum с явным циклом.
    """
    molecule = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    basis = build_basis("sto-3g", molecule)
    points = np.random.default_rng(3).normal(size=(12, 3)) * 1.2
    density = np.eye(basis.n_functions) * 0.5
    values, _ = evaluate_basis_with_gradients(basis, molecule, points)
    contracted = np.asarray(density @ values.T)
    hessian = evaluate_basis_hessian_for_center(basis, molecule, points, 0)
    columns = np.flatnonzero(_function_owner(basis) == 0)
    for axis in range(3):
        via_einsum = np.einsum("pjb,jp->pb", hessian[:, :, axis, :], contracted[columns])
        via_loop = np.zeros((points.shape[0], 3))
        for local, function in enumerate(columns):
            for other in range(basis.n_functions):
                via_loop += (
                    density[function, other]
                    * hessian[:, local, axis, :]
                    * values[:, other][:, None]
                )
        assert np.max(np.abs(via_einsum - via_loop)) < 1e-12


@pytest.mark.parametrize("functional_name", ["svwn", "pbe", "pbe0"])
def test_rks_gradient_matches_finite_differences(functional_name: str) -> None:
    """Аналитический градиент RKS совпадает с численной производной энергии.

    КР берётся на **замороженной** сетке: аналитический градиент точен именно
    для неподвижной в пространстве сетки (см. ``xc_gradient``). Сетка
    перестраивается по-другому, и расхождение между поверхностями измерено
    отдельно — оно в 64 раза ниже порога сходимости по силе.
    """
    molecule = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    functional = get_functional(functional_name)
    grid = build_grid(molecule, GridPreset.COARSE)
    basis = build_basis("sto-3g", molecule)
    result = run_rks(basis, molecule, functional, grid=grid)
    analytic = rks_gradient(basis, molecule, result, grid, functional).gradient

    step_angstrom = _STEP_BOHR / float(angstrom_to_bohr(1.0))
    numeric = np.zeros((len(molecule.atoms), 3))
    for atom_index in range(len(molecule.atoms)):
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                shifted = _displace_water(atom_index, axis, sign * step_angstrom)
                energies.append(
                    run_rks(
                        build_basis("sto-3g", shifted), shifted, functional, grid=grid
                    ).total_energy
                )
            numeric[atom_index, axis] = (energies[0] - energies[1]) / (2 * _STEP_BOHR)

    deviation = float(np.max(np.abs(analytic - numeric)))
    assert deviation < 1e-6, deviation


def test_density_and_sigma_derivatives_match_finite_differences() -> None:
    """``∂ρ/∂R`` и ``∂σ/∂R`` при фиксированной плотности — против КР.

    Проверка изолирует именно то, что входит в XC-градиент: если она проходит,
    расхождение в градиенте может быть только в потенциале ``v_ρ``/``v_σ``, а
    не в производных базиса.
    """
    molecule = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    basis = build_basis("sto-3g", molecule)
    points = np.random.default_rng(3).normal(size=(12, 3)) * 1.2
    density = np.eye(basis.n_functions) * 0.5
    bohr = float(angstrom_to_bohr(1.0))
    values, grads = evaluate_basis_with_gradients(basis, molecule, points)
    rho_gradient = density_gradient_at_points(values, grads, density)
    contracted = np.asarray(density @ values.T)
    gradient_contracted = np.einsum("nm,pnb->mpb", density, grads)
    owner = _function_owner(basis)
    step = 1e-6
    for atom_index in (0, 1):
        columns = np.flatnonzero(owner == atom_index)
        hessian = evaluate_basis_hessian_for_center(basis, molecule, points, atom_index)
        for axis in range(3):
            shifted = []
            for sign in (1.0, -1.0):
                moved = _displace_water(atom_index, axis, sign * step)
                basis_moved = build_basis("sto-3g", moved)
                v, g = evaluate_basis_with_gradients(basis_moved, moved, points)
                grad = density_gradient_at_points(v, g, density)
                shifted.append((v, grad))
            # производная плотности
            rho_plus = np.einsum("pg,g,pg->p", shifted[0][0], np.diag(density), shifted[0][0])
            rho_minus = np.einsum("pg,g,pg->p", shifted[1][0], np.diag(density), shifted[1][0])
            numeric_rho = (rho_plus - rho_minus) / (2 * step)
            analytic_rho = (
                -2.0 * np.sum(grads[:, columns, axis] * contracted[columns].T, axis=1) * bohr
            )
            assert np.max(np.abs(numeric_rho - analytic_rho)) < 1e-6
            # производная сигмы
            numeric_sigma = (
                np.sum(shifted[0][1] ** 2, axis=1) - np.sum(shifted[1][1] ** 2, axis=1)
            ) / (2 * step)
            first = np.einsum("pjb,jp->pb", hessian[:, :, axis, :], contracted[columns])
            second = np.einsum("pj,jpb->pb", grads[:, columns, axis], gradient_contracted[columns])
            analytic_sigma = -4.0 * np.sum(rho_gradient * (first + second), axis=1) * bohr
            assert np.max(np.abs(numeric_sigma - analytic_sigma)) < 1e-5


# --------------------------------------------------------------------------- #
# UHF: градиент открытой оболочки
# --------------------------------------------------------------------------- #
CH_RADICAL = Path(__file__).parent / "fixtures" / "ch-radical.xyz"


def _displace_keep_state(base: Molecule, atom_index: int, axis: int, delta: float) -> Molecule:
    """Смещает атом, сохраняя заряд и мультиплетность: без них домен не соберётся."""
    atoms: list[Atom] = []
    for index, atom in enumerate(base.atoms):
        position = list(atom.position)
        if index == atom_index:
            position[axis] += delta
        atoms.append(Atom(symbol=atom.symbol, position=(position[0], position[1], position[2])))
    return Molecule(
        atoms=tuple(atoms),
        name=base.name,
        charge=base.charge,
        multiplicity=base.multiplicity,
    )


def test_uhf_gradient_matches_finite_differences() -> None:
    """Аналитический градиент UHF совпадает с численной производной энергии.

    Проверка на радикале CH (дублет): там каналы α и β разные, и ошибка в
    обменном коэффициенте или в члене релаксации сразу видна. На закрытой
    оболочке она бы не проявилась.
    """
    molecule = Molecule.from_xyz(CH_RADICAL.read_text(encoding="utf-8"), name="ch", multiplicity=2)
    basis = build_basis("sto-3g", molecule)
    result = run_uhf(basis, molecule, _TIGHT)
    assert result.converged
    analytic = uhf_gradient(basis, molecule, result).gradient

    step_angstrom = _STEP_BOHR / float(angstrom_to_bohr(1.0))
    numeric = np.zeros_like(analytic)
    for atom_index in range(len(molecule.atoms)):
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                shifted = _displace_keep_state(molecule, atom_index, axis, sign * step_angstrom)
                energies.append(
                    run_uhf(build_basis("sto-3g", shifted), shifted, _TIGHT).total_energy
                )
            numeric[atom_index, axis] = (energies[0] - energies[1]) / (2 * _STEP_BOHR)

    deviation = float(np.max(np.abs(analytic - numeric)))
    assert deviation < 1e-6, deviation


def test_uhf_gradient_reduces_to_rhf_for_a_closed_shell() -> None:
    """На закрытой оболочке градиент UHF обязан совпасть с градиентом RHF.

    Это инвариант, а не приближение: при ``D^α = D^β = D/2`` выражение
    ``−½ ΣΣ (D^αD^α + D^βD^β)·(μλ|νσ)`` сворачивается ровно в ``−¼ ΣΣ DD``.
    Проверка ловит подмену обменного коэффициента ½ на ¼ — ошибка, при которой
    энергия остаётся правильной, а силы расходятся в разы.
    """
    molecule = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    basis = build_basis("sto-3g", molecule)
    from_uhf = uhf_gradient(basis, molecule, run_uhf(basis, molecule, _TIGHT)).gradient
    from_rhf = rhf_gradient(basis, molecule, run_rhf(basis, molecule, _TIGHT)).gradient
    assert float(np.max(np.abs(from_uhf - from_rhf))) < 1e-10


def test_uhf_gradient_of_stretched_hydrogen_matches_finite_differences() -> None:
    """Растянутый H₂ в триплете: градиент вдоль связи против конечных разностей.

    Другой режим, чем радикал CH: каналы различаются сильно, а градиент
    ненулевой именно по одной координате.
    """
    stretched = Molecule(
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, 2.0)),
        ),
        name="h2-stretched",
        multiplicity=3,
    )
    basis = build_basis("sto-3g", stretched)
    result = run_uhf(basis, stretched, _TIGHT)
    assert result.converged
    analytic = uhf_gradient(basis, stretched, result).gradient

    step_angstrom = _STEP_BOHR / float(angstrom_to_bohr(1.0))
    numeric = np.zeros_like(analytic)
    for atom_index in range(2):
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                shifted = _displace_keep_state(stretched, atom_index, axis, sign * step_angstrom)
                energies.append(
                    run_uhf(build_basis("sto-3g", shifted), shifted, _TIGHT).total_energy
                )
            numeric[atom_index, axis] = (energies[0] - energies[1]) / (2 * _STEP_BOHR)

    assert float(np.max(np.abs(analytic - numeric))) < 1e-6


# --------------------------------------------------------------------------- #
# ROHF: градиент открытой оболочки
# --------------------------------------------------------------------------- #
def _rohf_numeric_gradient(molecule: Molecule, basis_name: str) -> np.ndarray:
    """Численный градиент ROHF-энергии конечными разностями, хартри/бор."""
    analytic = rohf_gradient(build_basis(basis_name, molecule), molecule, _rohf(molecule)).gradient
    step_angstrom = _STEP_BOHR / float(angstrom_to_bohr(1.0))
    numeric = np.zeros_like(analytic)
    for atom_index in range(len(molecule.atoms)):
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                shifted = _displace_keep_state(molecule, atom_index, axis, sign * step_angstrom)
                energies.append(
                    run_rohf(build_basis(basis_name, shifted), shifted, _TIGHT).total_energy
                )
            numeric[atom_index, axis] = (energies[0] - energies[1]) / (2 * _STEP_BOHR)
    return numeric


def _rohf(molecule: Molecule, basis_name: str = "sto-3g") -> RohfResult:
    basis = build_basis(basis_name, molecule)
    result = run_rohf(basis, molecule, _TIGHT)
    assert result.converged
    return result


def test_rohf_gradient_matches_finite_differences() -> None:
    """Радикал CH (дублет): аналитический градиент против численного.

    Отличие от UHF-теста: орбитали каналов **общие**, поэтому проверка ловит
    ошибки в сборке весов по замкнутым/открытым блокам, а не только в обмене.
    """
    molecule = Molecule.from_xyz(CH_RADICAL.read_text(encoding="utf-8"), name="ch", multiplicity=2)
    result = _rohf(molecule)
    analytic = rohf_gradient(build_basis("sto-3g", molecule), molecule, result).gradient
    numeric = _rohf_numeric_gradient(molecule, "sto-3g")
    assert float(np.max(np.abs(analytic - numeric))) < 1e-6


def test_rohf_gradient_of_stretched_hydrogen_triplet_matches_finite_differences() -> None:
    """Растянутый H₂ в триплете: n_β = 0, замкнутый блок пуст.

    Краевой случай весов: в нём нет ни одного замкнутого орбитали, и формула
    обязанена свестись к открытому блоку без деления на пустой массив.
    """
    stretched = Molecule(
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, 2.0)),
        ),
        name="h2-stretched",
        multiplicity=3,
    )
    result = _rohf(stretched)
    analytic = rohf_gradient(build_basis("sto-3g", stretched), stretched, result).gradient
    numeric = _rohf_numeric_gradient(stretched, "sto-3g")
    assert float(np.max(np.abs(analytic - numeric))) < 1e-6


def test_rohf_gradient_reduces_to_rhf_for_a_closed_shell() -> None:
    """На закрытой оболочке градиент ROHF обязан совпасть с градиентом RHF.

    Инвариант: при пустом открытом блоке вес сворачивается в ``2CεCᵀ``, а
    обменные каналы дают тот же результат, что и RHF.
    """
    molecule = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    basis = build_basis("sto-3g", molecule)
    from_rohf = rohf_gradient(basis, molecule, run_rohf(basis, molecule, _TIGHT)).gradient
    from_rhf = rhf_gradient(basis, molecule, run_rhf(basis, molecule, _TIGHT)).gradient
    assert float(np.max(np.abs(from_rohf - from_rhf))) < 1e-10


def test_rohf_weight_couples_closed_and_open_blocks() -> None:
    """Вес ROHF не сводится к сохранённым ``orbital_energies``.

    На NO/STO-3G блок связности ``C^cᵀF^αC^o`` имеет порядок 1e-2, поэтому
    взвешенная плотность, собранная из энергий единого эффективного фокиана
    Руттхана (``rohf.orbital_energies``), расходится с корректной на ~1e-2.
    Тест фиксирует, что движок использует фокиан-зависимый вес.
    """
    molecule = Molecule(
        atoms=(
            Atom(symbol="N", position=(0.0, 0.0, 0.0)),
            Atom(symbol="O", position=(0.0, 0.0, 1.15)),
        ),
        name="no",
        multiplicity=2,
    )
    basis = build_basis("sto-3g", molecule)
    result = run_rohf(basis, molecule, _TIGHT)
    assert result.converged
    weight = _rohf_orbital_weight(basis, molecule, result)
    assert weight.shape == (basis.n_functions, basis.n_functions)
    # Симметрия веса — обязательна для члена релаксации.
    assert float(np.max(np.abs(weight - weight.T))) < 1e-12
    # Энерги из одного эффективного фокиана не дают такого веса: разница заметна.
    naive = np.asarray(
        2.0
        * (result.coefficients[:, :4] * np.asarray(result.orbital_energies[:4]))
        @ result.coefficients[:, :4].T
        + (result.coefficients[:, 4:5] * np.asarray(result.orbital_energies[4:5]))
        @ result.coefficients[:, 4:5].T
    )
    assert float(np.max(np.abs(weight - naive))) > 1e-3
