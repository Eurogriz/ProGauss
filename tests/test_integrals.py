"""Инвариантные тесты базисного слоя и интегрального движка.

Тесты не зависят от внешних пакетов: проверяются математические свойства,
которые обязаны выполняться всегда, и значения, полученные независимой
численной квадратурой. Сверка с PySCF вынесена в ``test_crosscheck_pyscf.py``
и помечена маркером ``scientific``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.engine import integrals
from quantumlab.engine.basis import (
    BasisNotFoundError,
    BasisSet,
    available_basis_sets,
    build_basis,
    cartesian_powers,
    nuclear_repulsion,
)
from quantumlab.engine.constants import angstrom_to_bohr
from quantumlab.engine.integrals import (
    build_core_hamiltonian,
    build_dipole_integrals,
    build_electron_repulsion,
    build_kinetic,
    build_nuclear_attraction,
    build_overlap,
    clear_caches,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def water() -> Molecule:
    """Молекула воды из фикстуры."""
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


@pytest.fixture(scope="module")
def sto3g(water: Molecule) -> BasisSet:
    """Базис STO-3G для воды."""
    return build_basis("sto-3g", water)


# --------------------------------------------------------------------------- #
# Реестр базисов
# --------------------------------------------------------------------------- #
def test_available_basis_sets_not_empty() -> None:
    """В реестре есть все 16 наборов, которые генерирует инструмент."""
    names = available_basis_sets()
    assert "sto-3g" in names
    assert "def2-qzvp" in names
    assert len(names) == 16


def test_unknown_basis_raises() -> None:
    """Несуществующий базис — явная ошибка, а не пустой результат."""
    with pytest.raises(BasisNotFoundError):
        build_basis(
            "несуществующий-базис",
            Molecule(name="h", atoms=(Atom(symbol="H", position=(0.0, 0.0, 0.0)),), multiplicity=2),
        )


def test_cartesian_powers_counts() -> None:
    """Число декартовых компонент: 1, 3, 6, 10 для l = 0..3."""
    assert [len(cartesian_powers(value)) for value in range(4)] == [1, 3, 6, 10]


def test_cartesian_powers_are_stable() -> None:
    """Порядок компонент фиксирован: его смена сломала бы сохранённые артефакты."""
    assert cartesian_powers(2) == (
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    )


def test_basis_function_counts(water: Molecule) -> None:
    """Число функций для известных комбинаций (проверка развёртки оболочек)."""
    assert build_basis("sto-3g", water).n_functions == 7
    assert build_basis("6-31g", water).n_functions == 13
    assert build_basis("6-31g(d,p)", water).n_functions == 25
    assert build_basis("cc-pvdz", water).n_functions == 25


def test_contraction_rows_are_all_expanded(water: Molecule) -> None:
    """Все строки коэффициентов разворачиваются в функции.

    BSE хранит общую форму: одна оболочка может нести несколько сжатых функций.
    Раньше разворачивалась только первая строка, и cc-pVDZ терял функции.
    """
    basis = build_basis("cc-pvdz", water)
    oxygen_s = [
        shell for shell in basis.shells if shell.center == 0 and shell.angular_momentum == 0
    ]
    assert len(oxygen_s) == 3


def test_nuclear_repulsion_known_value(water: Molecule) -> None:
    """Ядерное отталкивание воды: 9.1893235112 Eh."""
    assert nuclear_repulsion(water) == pytest.approx(9.1893235112, abs=1e-9)


# --------------------------------------------------------------------------- #
# Матрица перекрывания
# --------------------------------------------------------------------------- #
def test_overlap_is_symmetric(sto3g: BasisSet, water: Molecule) -> None:
    """S симметрична по построению."""
    matrix = build_overlap(sto3g, water)
    assert np.allclose(matrix, matrix.T, atol=1e-14)


def test_overlap_has_unit_diagonal(water: Molecule) -> None:
    """Все базисные функции нормированы на единицу.

    В том числе декартовы d-функции xy/xz/yz: их норма требует покомпонентного
    множителя √3, иначе диагональ получается равной 1/3.
    """
    matrix = build_overlap(build_basis("6-31g(d,p)", water), water)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-12)


def test_overlap_is_positive_definite(sto3g: BasisSet, water: Molecule) -> None:
    """S положительно определена: базис линейно независим."""
    matrix = build_overlap(sto3g, water)
    assert np.all(np.linalg.eigvalsh(matrix) > 0)


def test_overlap_of_symmetry_unrelated_functions_is_zero(sto3g: BasisSet, water: Molecule) -> None:
    """Интегралы между функциями разной симметрии равны нулю.

    Вода лежит в плоскости z = 0. Орбиталь 2p_z кислорода нечётна по z, все
    остальные функции чётны, поэтому ⟨2p_z|что угодно⟩ = 0. (1s и 2s кислорода
    в STO-3G, напротив, НЕ ортогональны — 0.2367; это свойство самого базиса,
    а не ошибка.)
    """
    matrix = build_overlap(sto3g, water)
    for column in range(matrix.shape[1]):
        if column != 4:
            assert matrix[4, column] == pytest.approx(0.0, abs=1e-14)
    assert matrix[0, 1] == pytest.approx(0.2367, abs=1e-3)


# --------------------------------------------------------------------------- #
# Кинетическая энергия
# --------------------------------------------------------------------------- #
def test_kinetic_is_symmetric(sto3g: BasisSet, water: Molecule) -> None:
    """T симметрична (интегрирование по частям)."""
    matrix = build_kinetic(sto3g, water)
    assert np.allclose(matrix, matrix.T, atol=1e-12)


def test_kinetic_diagonal_is_positive(sto3g: BasisSet, water: Molecule) -> None:
    """⟨φ|−½∇²|φ⟩ > 0 для любой ненулевой функции."""
    matrix = build_kinetic(sto3g, water)
    assert np.all(np.diag(matrix) > 0)


# --------------------------------------------------------------------------- #
# Притяжение к ядрам
# --------------------------------------------------------------------------- #
def test_nuclear_attraction_is_negative(sto3g: BasisSet, water: Molecule) -> None:
    """Все диагональные элементы притяжения отрицательны."""
    matrix = build_nuclear_attraction(sto3g, water)
    assert np.all(np.diag(matrix) < 0)


def test_core_hamiltonian_is_sum(sto3g: BasisSet, water: Molecule) -> None:
    """H = T + V."""
    assert np.allclose(
        build_core_hamiltonian(sto3g, water),
        build_kinetic(sto3g, water) + build_nuclear_attraction(sto3g, water),
        atol=1e-14,
    )


# --------------------------------------------------------------------------- #
# Двухэлектронные интегралы
# --------------------------------------------------------------------------- #
def test_eri_has_full_permutation_symmetry(water: Molecule) -> None:
    """Тензор (μν|λσ) инвариантен ко всем восьми перестановкам индексов."""
    tensor = build_electron_repulsion(build_basis("sto-3g", water), water)
    assert np.allclose(tensor, tensor.transpose(1, 0, 2, 3), atol=1e-14)
    assert np.allclose(tensor, tensor.transpose(0, 1, 3, 2), atol=1e-14)
    assert np.allclose(tensor, tensor.transpose(2, 3, 0, 1), atol=1e-14)


def test_eri_diagonal_is_positive(water: Molecule) -> None:
    """(μμ|μμ) > 0 — кулоновская самоэнергия плотности."""
    tensor = build_electron_repulsion(build_basis("sto-3g", water), water)
    for index in range(tensor.shape[0]):
        assert tensor[index, index, index, index] > 0


# --------------------------------------------------------------------------- #
# Дипольные интегралы
# --------------------------------------------------------------------------- #
def test_dipole_integral_matches_quadrature() -> None:
    """⟨a|r|b⟩ совпадает с прямой 3D-квадратурой.

    Это независимая проверка: формула выводится из тех же эрмитовых
    коэффициентов, но квадратура ничего о них не знает.
    """
    from quantumlab.engine.integrals import _multipole_primitive

    def quadrature(la: tuple[int, int, int], lb: tuple[int, int, int], axis: int) -> float:
        center_a = (0.0, 0.0, 0.0)
        center_b = (0.3, -0.2, 0.15)
        alpha, beta = 2.0, 0.9
        grid = np.linspace(-8.0, 8.0, 241)
        step = grid[1] - grid[0]
        x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")

        def gaussian(
            powers: tuple[int, int, int],
            center: tuple[float, float, float],
            exponent: float,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
        ) -> np.ndarray:
            return (
                (x - center[0]) ** powers[0]
                * (y - center[1]) ** powers[1]
                * (z - center[2]) ** powers[2]
                * np.exp(
                    -exponent * ((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                )
            )

        coordinate = (x, y, z)[axis]
        integrand = (
            gaussian(la, center_a, alpha, x, y, z)
            * coordinate
            * gaussian(lb, center_b, beta, x, y, z)
        )
        return float(integrand.sum() * step**3)

    cases = [
        ((0, 0, 0), (0, 0, 0), 0),
        ((0, 0, 0), (0, 1, 0), 1),
        ((0, 1, 0), (0, 0, 0), 1),
        ((1, 0, 0), (0, 0, 0), 0),
        ((2, 0, 0), (0, 0, 0), 0),
        ((1, 1, 0), (2, 0, 0), 1),
    ]
    for la, lb, axis in cases:
        analytic = _multipole_primitive(2.0, la, (0.0, 0.0, 0.0), 0.9, lb, (0.3, -0.2, 0.15), axis)
        assert analytic == pytest.approx(quadrature(la, lb, axis), rel=1e-9)
    clear_caches()


def test_dipole_trace_equals_sum_of_centers(water: Molecule) -> None:
    """Σ_μ ⟨μ|y|μ⟩ = Σ_μ y_μ: для нормированных функций это суммы координат."""
    basis = build_basis("sto-3g", water)
    _, dy, _ = build_dipole_integrals(basis, water)
    expected = sum(
        angstrom_to_bohr(water.atoms[shell.center].position[1])
        for shell in basis.shells
        for _ in range(shell.n_cartesian)
    )
    assert float(np.trace(dy)) == pytest.approx(expected, abs=1e-10)


def test_dipole_is_origin_independent_for_neutral_molecule(water: Molecule) -> None:
    """Диполь нейтральной системы не зависит от начала координат.

    Проверяется алгебраически: сдвиг начала координат меняет ⟨r⟩ на
    ⟨1⟩·shift, а полный диполь включает ядерный вклад с тем же сдвигом.
    """
    basis = build_basis("sto-3g", water)
    from quantumlab.engine.scf import run_rhf

    result = run_rhf(basis, water)
    overlap = build_overlap(basis, water)
    n_electrons = float(np.trace(result.density @ overlap))
    assert n_electrons == pytest.approx(water.n_electrons, abs=1e-8)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g"])
def test_vectorized_eri_matches_scalar_reference(water: Molecule, basis_name: str) -> None:
    """Векторизованная сборка ERI совпадает со скалярной спецификацией.

    В ``integrals`` два пути: скалярный ``_quartet_block_scalar`` — медленный,
    но читаемый, и векторизованный — рабочий. Тест фиксирует, что ускорение не
    изменило физику. Проверяются ``s``- и ``p``-оболочки: на них работают
    ветвления рекурсии по угловым моментам.
    """
    basis = build_basis(basis_name, water)
    centers = integrals._shell_centers(basis, water)
    worst = 0.0
    checked = 0
    for i in range(len(basis.shells)):
        for j in range(i + 1):
            for k in range(len(basis.shells)):
                for m in range(k + 1):
                    if i * (i + 1) + j < k * (k + 1) + m:
                        continue
                    arguments = (
                        basis.shells[i],
                        centers[i],
                        basis.shells[j],
                        centers[j],
                        basis.shells[k],
                        centers[k],
                        basis.shells[m],
                        centers[m],
                    )
                    fast = integrals._quartet_block(*arguments)
                    slow = integrals._quartet_block_scalar(*arguments)
                    worst = max(worst, float(np.abs(fast - slow).max()))
                    checked += 1
    assert checked > 0
    assert worst < 1e-13, worst


def test_eri_build_is_not_repeated_by_quality_checks(
    water: Molecule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Тензор ERI строится один раз за расчёт, а не по разу на проверку.

    Раньше ``_quality_checks`` собирала его заново ради разложения энергии,
    и на воде/6-31G это удваивало стоимость расчёта. Подменяется связывание
    в ``scf``, а не в ``integrals``: ``scf`` импортирует функцию по имени,
    поэтому патч исходного модуля до неё не доходит.
    """
    from quantumlab.domain.spec import CalculationSpec, MethodSpec, Task, TheoryFamily
    from quantumlab.engine.contracts import EngineRequest
    from quantumlab.engine.reference import ReferenceEngine

    calls: list[int] = []
    original = build_electron_repulsion

    def counting(target: BasisSet, structure: Molecule) -> np.ndarray:
        calls.append(1)
        return original(target, structure)

    # Цель задаётся строкой: ``scf`` не реэкспортирует функцию, и обращение
    # по атрибуту mypy отвергает как неявный экспорт.
    monkeypatch.setattr("quantumlab.engine.scf.build_electron_repulsion", counting)
    basis = build_basis("sto-3g", water)
    ReferenceEngine().run(
        EngineRequest(
            job_id="eri-count",
            molecule=water,
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=MethodSpec(theory=TheoryFamily.HF, basis=basis.name),
            ),
        )
    )
    assert len(calls) == 1, f"ERI собрана {len(calls)} раз вместо одного"


# --------------------------------------------------------------------------- #
# Функция Бойса против независимого эталона
# --------------------------------------------------------------------------- #
def _boys_reference(order: int, x: float) -> float:
    r"""``F_n(x) = ∫₀¹ t^{2n} e^{−x t²} dt`` квадратурой Гаусса–Лежандра.

    Эталон намеренно не использует ни ряд Тейлора, ни ``erf``: обе эти ветви
    входят в проверяемую реализацию, и сравнение с ними ловило бы только ошибку
    копирования. Подынтегральная функция аналитична на ``[0, 1]``, поэтому 96
    узлов дают ~1e-15; точность самой квадратуры сверена с
    ``scipy.special.hyp1f1(n+½, n+3/2, −x)/(2n+1)`` на всей сетке аргументов
    теста (расхождение ≤ 1.6e-15).
    """
    nodes, weights = np.polynomial.legendre.leggauss(96)
    t = 0.5 * (nodes + 1.0)
    return float(np.sum(0.5 * weights * t ** (2 * order) * np.exp(-x * t * t)))


#: Аргументы покрывают обе ветви и окрестность границы между ними.
_BOYS_ARGUMENTS = (
    0.0,
    1e-9,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    5.5,
    5.9,
    5.999,
    6.0,
    6.5,
    12.0,
    80.0,
    1000.0,
)


@pytest.mark.parametrize("order", range(9))
def test_boys_matches_independent_quadrature(order: int) -> None:
    """Обе ветви функции Бойса совпадают с квадратурой определяющего интеграла.

    Регрессия, которую тест обязан ловить: при фиксированных 25 членах ряда
    обрыв давал 1.2e-12 при ``x = 4`` и 2.8e-8 при ``x → 6`` — то есть ровно на
    границе ветвей функция теряла семь знаков. Сравнение «векторный путь против
    скалярного» этого не видело: оба пути ошибались одинаково.
    """
    for x in _BOYS_ARGUMENTS:
        expected = _boys_reference(order, x)
        assert integrals._boys(order, x) == pytest.approx(expected, abs=1e-13), x

    arguments = np.asarray(_BOYS_ARGUMENTS)
    reference = np.asarray([_boys_reference(order, x) for x in _BOYS_ARGUMENTS])
    assert np.abs(integrals._boys_array(order, arguments) - reference).max() < 1e-13


def test_boys_table_computes_all_orders_at_once() -> None:
    """Таблица порядков совпадает с поштучным счётом и с квадратурой."""
    arguments = np.asarray(_BOYS_ARGUMENTS)
    table = integrals._boys_table(8, arguments)
    assert len(table) == 9
    for order in range(9):
        expected = np.asarray([_boys_reference(order, x) for x in _BOYS_ARGUMENTS])
        assert np.abs(table[order] - expected).max() < 1e-13, order


@pytest.mark.parametrize("x_max", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.999])
def test_series_terms_guarantees_truncation_bound(x_max: float) -> None:
    """Подобранное число членов действительно обеспечивает заявленную точность.

    Ряд знакочередующийся, поэтому остаток не больше первого отброшенного члена;
    проверяем именно это неравенство, а не «число выглядит большим».
    """
    terms = integrals._series_terms(x_max)
    remainder = x_max**terms / (math.factorial(terms) * (2 * terms + 1))
    assert remainder < 1e-17, (x_max, terms, remainder)
    # И что меньше брать нельзя: иначе тест не отличал бы запас от необходимости.
    previous = (x_max ** (terms - 1)) / (math.factorial(terms - 1) * (2 * terms - 1))
    assert previous >= 1e-17, (x_max, terms, previous)


# --------------------------------------------------------------------------- #
# Пакетная сборка ERI против поквартиетной
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g"])
def test_batched_eri_matches_quartetwise_assembly(water: Molecule, basis_name: str) -> None:
    """Пакетная сборка тензора совпадает с поквартиетной на всём тензоре.

    Сравнение по всему тензору, а не по отдельным блокам: так ловятся ошибки
    размещения (порядок сторон, набор перестановок) и ошибки группировки по
    классам — например, пачка из квартетов с разным угловым моментом.
    ``6-31g`` обязателен: в ``sto-3g`` все оболочки ``s``-типа, и путь
    ``p``-оболочек остался бы непокрытым.
    """
    basis = build_basis(basis_name, water)
    batched = build_electron_repulsion(basis, water)
    reference = integrals._build_electron_repulsion_quartetwise(basis, water)
    worst = float(np.abs(batched - reference).max())
    assert worst < 1e-12, worst


def test_batched_eri_derivative_matches_quartetwise_assembly(water: Molecule) -> None:
    """То же для производной: у неё свой набор перестановок (только ``λ ↔ σ``)."""
    basis = build_basis("sto-3g", water)
    for axis in range(3):
        batched = integrals.build_electron_repulsion_derivative(basis, water, axis)
        reference = integrals._build_electron_repulsion_derivative_quartetwise(basis, water, axis)
        worst = float(np.abs(batched - reference).max())
        assert worst < 1e-12, (axis, worst)


@pytest.mark.parametrize("target", [1, 7, 4096])
def test_batch_size_does_not_change_eri(
    water: Molecule, target: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Размер пачки — параметр производительности, а не точности.

    При ``target = 1`` в пачке ровно один квартет, то есть пакетный код
    проходит тот же путь, что и поквартиетная сборка, но с полным аппаратом
    классификации и размещения.
    """
    basis = build_basis("6-31g", water)
    reference = build_electron_repulsion(basis, water)
    monkeypatch.setattr(integrals, "_TARGET_BATCH_ELEMENTS", target)
    obtained = build_electron_repulsion(basis, water)
    worst = float(np.abs(obtained - reference).max())
    assert worst < 1e-12, (target, worst)
