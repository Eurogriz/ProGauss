"""Сверка референсного движка с PySCF как независимым оракулом.

Согласно ADR-002 внешние пакеты используются **только для проверки**, никогда
как источник истины. PySCF — широко верифицированная реализация; совпадение с
ним до 1e-7 доказывает корректность наших интегралов и SCF.

Сравнивать можно только величины, инвариантные относительно выбора базисных
функций внутри одного и того же пространства:

* полную энергию, орбитальные энергии, дипольный момент;
* спектр матрицы перекрывания, приведённой к единичной диагонали, — и только
  для тех базисов, где наборы сжатых функций совпадают буквально.

Энергии совпадают и для cc-pV*: BSE хранит их в общей форме (один примитив
может входить в несколько сжатий), а PySCF — в сегментированной переразвёртке.
Пространства одинаковы, но сами функции — разные линейные комбинации, поэтому
матрица перекрывания (и её спектр) у них различается, хотя энергия нет.

Тесты помечены ``scientific`` и пропускаются, если PySCF не установлен.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import GridPreset
from quantumlab.engine.basis import build_basis
from quantumlab.engine.constants import angstrom_to_bohr
from quantumlab.engine.dft import run_rks
from quantumlab.engine.functional import get_functional
from quantumlab.engine.gradients import rks_gradient, rohf_gradient, uhf_gradient
from quantumlab.engine.integrals import build_dipole_integrals, build_overlap
from quantumlab.engine.quadrature import build_grid
from quantumlab.engine.scf import ScfSettings, run_rhf, run_rohf, run_uhf

pyscf = pytest.importorskip("pyscf", reason="PySCF нужен только для независимой сверки")

#: ``importorskip`` тянет только верхний пакет: ``pyscf.gto`` и ``pyscf.scf`` подхватываются
#: как побочный эффект, а ``pyscf.dft`` — нет. Статический ``import pyscf.dft``
#: заставил бы mypy искать у PySCF заглушки типов, которых нет, поэтому
#: подмодуль регистрируется через ``importlib``.
importlib.import_module("pyscf.dft")

pytestmark = pytest.mark.scientific

FIXTURES = Path(__file__).parent / "fixtures"
TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)

#: Наши имена базисов и соответствующие им имена в PySCF. Все расчёты идут в
#: декартовой схеме (``cart=True``), потому что наш движок декартов.
CARTESIAN_PAIRS = [
    ("sto-3g", "STO-3G"),
    ("6-31g", "6-31G"),
    ("6-31g(d,p)", "6-31G**"),
    ("cc-pvdz", "cc-pVDZ"),
    ("def2-svp", "def2-SVP"),
]

#: Базисы, у которых наборы сжатых функций буквально совпадают с PySCF, —
#: для них корректно сравнивать ещё и спектр нормированной матрицы S.
SAME_CONTRACTION_PAIRS = [("sto-3g", "STO-3G"), ("6-31g", "6-31G"), ("6-31g(d,p)", "6-31G**")]


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def _pyscf_molecule(molecule: Molecule, basis: str) -> Any:
    """Собирает ту же молекулу в PySCF. Атомы разделяются ';', не пробелом."""
    atom_string = "; ".join(
        f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}"
        for atom in molecule.atoms
    )
    return pyscf.gto.M(
        atom=atom_string,
        basis=basis,
        cart=True,
        spin=molecule.multiplicity - 1,
        charge=molecule.charge,
        unit="Angstrom",
        verbose=0,
    )


@pytest.mark.parametrize(("ours", "theirs"), CARTESIAN_PAIRS)
def test_total_energy_matches_pyscf(ours: str, theirs: str) -> None:
    """Полная RHF-энергия совпадает с PySCF в декартовой схеме."""
    molecule = _water()
    result = run_rhf(build_basis(ours, molecule), molecule, TIGHT)
    assert result.converged, f"наш SCF не сошёлся на {ours}"
    reference = (
        _pyscf_molecule(molecule, theirs).RHF().run(conv_tol=1e-12, max_cycle=300, verbose=0)
    )
    assert result.total_energy == pytest.approx(reference.e_tot, abs=1e-6)


@pytest.mark.parametrize(("ours", "theirs"), CARTESIAN_PAIRS)
def test_orbital_energies_match_pyscf(ours: str, theirs: str) -> None:
    """Отсортированные орбитальные энергии совпадают (порядок функций может отличаться)."""
    molecule = _water()
    result = run_rhf(build_basis(ours, molecule), molecule, TIGHT)
    reference = (
        _pyscf_molecule(molecule, theirs).RHF().run(conv_tol=1e-12, max_cycle=300, verbose=0)
    )
    ours_sorted = np.sort(result.orbital_energies)
    theirs_sorted = np.sort(reference.mo_energy)
    assert ours_sorted.shape == theirs_sorted.shape, "разное число базисных функций"
    assert np.allclose(ours_sorted, theirs_sorted, atol=1e-5)


@pytest.mark.parametrize(("ours", "theirs"), CARTESIAN_PAIRS)
def test_dipole_moment_matches_pyscf(ours: str, theirs: str) -> None:
    """Дипольный момент совпадает; величина инвариантна к началу координат."""
    molecule = _water()
    basis = build_basis(ours, molecule)
    result = run_rhf(basis, molecule, TIGHT)
    reference = (
        _pyscf_molecule(molecule, theirs).RHF().run(conv_tol=1e-12, max_cycle=300, verbose=0)
    )

    dx, dy, dz = build_dipole_integrals(basis, molecule)
    nuclear = sum(
        atom.z * np.array([angstrom_to_bohr(value) for value in atom.position])
        for atom in molecule.atoms
    )
    electronic = np.array([float(np.sum(result.density * axis)) for axis in (dx, dy, dz)])
    ours_dipole = nuclear - electronic
    assert np.allclose(ours_dipole, np.asarray(reference.dip_moment(unit="AU")), atol=1e-5)


@pytest.mark.parametrize(("ours", "theirs"), SAME_CONTRACTION_PAIRS)
def test_overlap_spectrum_matches_pyscf(ours: str, theirs: str) -> None:
    """Спектр S, приведённой к единичной диагонали, совпадает.

    Нормировка обязательна: у PySCF своя нормировка декартовых d-функций,
    а спектр S не инвариантен относительно масштабирования базиса.
    """
    molecule = _water()
    basis = build_basis(ours, molecule)
    reference_molecule = _pyscf_molecule(molecule, theirs)

    def normalized_spectrum(matrix: np.ndarray) -> np.ndarray:
        root = np.sqrt(np.diag(matrix))
        return np.sort(np.linalg.eigvalsh(matrix / np.outer(root, root)))

    ours_spectrum = normalized_spectrum(build_overlap(basis, molecule))
    theirs_spectrum = normalized_spectrum(reference_molecule.intor("int1e_ovlp"))
    assert ours_spectrum.shape == theirs_spectrum.shape
    assert np.allclose(ours_spectrum, theirs_spectrum, atol=1e-6)


def test_overlap_matrix_matches_pyscf_elementwise_for_sto3g() -> None:
    """Для STO-3G порядок базисных функций совпадает с PySCF — сравнение поэлементное.

    Это самый строгий из доступных тестов: он проверяет не только спектр, но и
    каждую матричную элемент, включая порядок декартовых компонент.
    """
    molecule = _water()
    basis = build_basis("sto-3g", molecule)
    reference = _pyscf_molecule(molecule, "STO-3G")
    ours = build_overlap(basis, molecule)
    theirs = reference.intor("int1e_ovlp")
    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs, atol=1e-7)


# --------------------------------------------------------------------------- #
# UHF: открытая оболочка
# --------------------------------------------------------------------------- #
#: Системы с открытой оболочкой: (имя, атомы, заряд, мультиплетность).
#: Подобраны так, чтобы покрыть разные случаи: один электрон, катион-радикал,
#: многоэлектронный радикал и растянутая связь.
OPEN_SHELL_CASES = [
    ("hydrogen-atom", [("H", (0.0, 0.0, 0.0))], 0, 2),
    ("hydrogen-cation", [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))], 1, 2),
    ("ch-radical", [("C", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.117))], 0, 2),
]


@pytest.mark.parametrize(("name", "atoms", "charge", "multiplicity"), OPEN_SHELL_CASES)
def test_uhf_energy_matches_pyscf(
    name: str, atoms: list[tuple[str, tuple[float, float, float]]], charge: int, multiplicity: int
) -> None:
    """Энергия UHF совпадает с PySCF — независимое подтверждение фокиана.

    Формулы UHF отличаются от RHF только построением фокиана (обмен по своему
    каналу, кулон по полной плотности), поэтому сверка энергии проверяет именно
    их, а не интегралы: те уже подтверждены RHF-тестами выше.
    """
    molecule = Molecule.from_atoms(
        [symbol for symbol, _ in atoms],
        [position for _, position in atoms],
        charge=charge,
        multiplicity=multiplicity,
        name=name,
    )
    basis = build_basis("sto-3g", molecule)
    ours = run_uhf(basis, molecule, TIGHT)
    assert ours.converged

    mol = pyscf.gto.M(
        atom=[[symbol, list(position)] for symbol, position in atoms],
        basis="sto-3g",
        spin=multiplicity - 1,
        charge=charge,
        cart=True,
        verbose=0,
    )
    reference = pyscf.scf.UHF(mol).run(conv_tol=1e-12)
    assert ours.total_energy == pytest.approx(reference.e_tot, abs=1e-7)


@pytest.mark.parametrize(("name", "atoms", "charge", "multiplicity"), OPEN_SHELL_CASES)
def test_uhf_spin_expectation_matches_pyscf(
    name: str, atoms: list[tuple[str, tuple[float, float, float]]], charge: int, multiplicity: int
) -> None:
    """⟨Ŝ²⟩ совпадает с PySCF, включая ненулевое спиновое загрязнение.

    Проверка важна именно на радикалах: у CH загрязнение не равно нулю, и
    совпадение такого значения подтверждает, что формула учитывает перекрытия
    орбиталей разных каналов, а не считает их ортогональными.
    """
    molecule = Molecule.from_atoms(
        [symbol for symbol, _ in atoms],
        [position for _, position in atoms],
        charge=charge,
        multiplicity=multiplicity,
        name=name,
    )
    basis = build_basis("sto-3g", molecule)
    ours = run_uhf(basis, molecule, TIGHT)

    mol = pyscf.gto.M(
        atom=[[symbol, list(position)] for symbol, position in atoms],
        basis="sto-3g",
        spin=multiplicity - 1,
        charge=charge,
        cart=True,
        verbose=0,
    )
    reference = pyscf.scf.UHF(mol).run(conv_tol=1e-12)
    assert ours.s_squared == pytest.approx(reference.spin_square()[0], abs=1e-6)


def test_uhf_orbital_energies_match_pyscf() -> None:
    """Энергии орбиталей обоих каналов совпадают с PySCF."""
    molecule = Molecule.from_atoms(
        ["C", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.117)], multiplicity=2, name="ch"
    )
    basis = build_basis("sto-3g", molecule)
    ours = run_uhf(basis, molecule, TIGHT)

    mol = pyscf.gto.M(
        atom=[["C", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.117]]],
        basis="sto-3g",
        spin=1,
        cart=True,
        verbose=0,
    )
    reference = pyscf.scf.UHF(mol).run(conv_tol=1e-12)
    np.testing.assert_allclose(
        np.sort(ours.alpha_energies), np.sort(reference.mo_energy[0]), atol=1e-6
    )
    np.testing.assert_allclose(
        np.sort(ours.beta_energies), np.sort(reference.mo_energy[1]), atol=1e-6
    )


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [("svwn", "SVWN"), ("pbe", "PBE"), ("pbe0", "PBE0")],
)
def test_dft_nuclear_gradient_matches_pyscf(ours: str, theirs: str) -> None:
    """Аналитический ядерный градиент RKS против независимого оракула.

    Конечные разности проверяют градиент против нашей же энергии: они ловят
    ошибку в формуле, но не в самой энергии. PySCF считает градиент своим кодом
    и своей сеткой, поэтому совпадение до ~1e-06 (разница квадратур) подтверждает
    и формулу, и её реализацию.

    ``grid_response=False`` у PySCF соответствует нашей модели: сетка считается
    неподвижной в пространстве. При ``True`` расхождение растёт лишь на
    2.5e-08 — 3.8e-08, то есть отклик весов Беке мал по сравнению с разницей
    квадратурных сеток.
    """
    molecule = _water()
    functional = get_functional(ours)
    grid = build_grid(molecule, GridPreset.ULTRAFINE)
    basis = build_basis("sto-3g", molecule)
    result = run_rks(basis, molecule, functional, grid=grid)
    ours_gradient = rks_gradient(basis, molecule, result, grid, functional).gradient

    reference = pyscf.dft.RKS(_pyscf_molecule(molecule, "STO-3G"))
    reference.xc = theirs
    reference.grids.atom_grid = (120, 974)
    reference.conv_tol = 1e-12
    reference.run()
    gradient = reference.nuc_grad_method()
    gradient.grid_response = False
    theirs_gradient = np.asarray(gradient.kernel())

    deviation = float(np.max(np.abs(ours_gradient - theirs_gradient)))
    assert deviation < 5e-6, deviation


def test_uhf_nuclear_gradient_matches_pyscf() -> None:
    """Ядерный градиент UHF против независимого оракула.

    Радикал CH (дублет): каналы α и β разные, поэтому проверка покрывает именно
    открытую оболочку. На закрытой оболочке градиент UHF совпал бы с RHF и
    ошибка в обменном коэффициенте осталась бы незамеченной.
    """
    molecule = Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), name="ch", multiplicity=2
    )
    basis = build_basis("sto-3g", molecule)
    result = run_uhf(basis, molecule, TIGHT)
    ours_gradient = uhf_gradient(basis, molecule, result).gradient

    reference = pyscf.scf.UHF(_pyscf_molecule(molecule, "STO-3G")).run(conv_tol=1e-12)
    theirs_gradient = np.asarray(reference.nuc_grad_method().kernel())

    deviation = float(np.max(np.abs(ours_gradient - theirs_gradient)))
    assert deviation < 1e-6, deviation


# --------------------------------------------------------------------------- #
# ROHF: открытая оболочка с общими орбиталями каналов
# --------------------------------------------------------------------------- #
#: Системы для сверки ROHF. Подобраны, чтобы покрыть всю структуру весов:
#: дублеты с разным числом замкнутых оболочек (включая NO, где блок связности
#: ``C^cᵀF^αC^o`` заметен), триплеты и краевой случай с пустым замкнутым блоком.
ROHF_CASES = [
    ("hydrogen-atom", [("H", (0.0, 0.0, 0.0))], 0, 2),
    ("ch-doublet", [("C", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.117))], 0, 2),
    ("oh-doublet", [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.97))], 0, 2),
    ("no-doublet", [("N", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.15))], 0, 2),
    ("h2-triplet", [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.0))], 0, 3),
    ("o2-triplet", [("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.207))], 0, 3),
]


def _rohf_molecule(
    name: str,
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    multiplicity: int,
) -> Molecule:
    return Molecule.from_atoms(
        [symbol for symbol, _ in atoms],
        [position for _, position in atoms],
        charge=charge,
        multiplicity=multiplicity,
        name=name,
    )


@pytest.mark.parametrize(("name", "atoms", "charge", "multiplicity"), ROHF_CASES)
def test_rohf_energy_matches_pyscf(
    name: str,
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    multiplicity: int,
) -> None:
    """Энергия ROHF совпадает с PySCF.

    ROHF отличается от UHF построением плотности (общие орбитали каналов),
    поэтому сверка энергии подтверждает именно этот фокиан, а не переиспользует
    проверку UHF.
    """
    molecule = _rohf_molecule(name, atoms, charge, multiplicity)
    ours = run_rohf(build_basis("sto-3g", molecule), molecule, TIGHT)
    assert ours.converged

    mol = pyscf.gto.M(
        atom=[[symbol, list(position)] for symbol, position in atoms],
        basis="sto-3g",
        spin=multiplicity - 1,
        charge=charge,
        cart=True,
        verbose=0,
    )
    reference = pyscf.scf.ROHF(mol).run(conv_tol=1e-12)
    assert ours.total_energy == pytest.approx(reference.e_tot, abs=1e-7)


@pytest.mark.parametrize(("name", "atoms", "charge", "multiplicity"), ROHF_CASES)
def test_rohf_nuclear_gradient_matches_pyscf(
    name: str,
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    multiplicity: int,
) -> None:
    """Ядерный градиент ROHF против независимого оракула.

    Самая чувствительная проверка: член релаксации строится из фокиан-зависимых
    весов (замкнутый блок по ``Fc``, открытый по ``F^α``, плюс блок связности).
    Блок связности заметен именно на NO, поэтому он обязателен в выборке.
    """
    molecule = _rohf_molecule(name, atoms, charge, multiplicity)
    basis = build_basis("sto-3g", molecule)
    result = run_rohf(basis, molecule, TIGHT)
    ours_gradient = rohf_gradient(basis, molecule, result).gradient

    mol = pyscf.gto.M(
        atom=[[symbol, list(position)] for symbol, position in atoms],
        basis="sto-3g",
        spin=multiplicity - 1,
        charge=charge,
        cart=True,
        verbose=0,
    )
    reference = pyscf.scf.ROHF(mol).run(conv_tol=1e-12)
    theirs_gradient = np.asarray(reference.nuc_grad_method().kernel())

    deviation = float(np.max(np.abs(ours_gradient - theirs_gradient)))
    assert deviation < 1e-6, deviation


# --------------------------------------------------------------------------- #
# ROHF: открытая оболочка с общими орбиталями каналов
# --------------------------------------------------------------------------- #
#: Системы для сверки ROHF. Подобраны, чтобы покрыть всю структуру весов:
#: дублеты с разным числом замкнутых оболочек (включая NO, где блок связности
#: ``C^cᵀF^αC^o`` заметен), триплеты и краевой случай с пустым замкнутым блоком.
ROHF_CASES = [
    ("hydrogen-atom", [("H", (0.0, 0.0, 0.0))], 0, 2),
    ("ch-doublet", [("C", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.117))], 0, 2),
    ("oh-doublet", [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.97))], 0, 2),
    ("no-doublet", [("N", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.15))], 0, 2),
    ("h2-triplet", [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.0))], 0, 3),
    ("o2-triplet", [("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.207))], 0, 3),
]


def _rohf_molecule(
    name: str,
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    multiplicity: int,
) -> Molecule:
    return Molecule.from_atoms(
        [symbol for symbol, _ in atoms],
        [position for _, position in atoms],
        charge=charge,
        multiplicity=multiplicity,
        name=name,
    )


@pytest.mark.parametrize(("name", "atoms", "charge", "multiplicity"), ROHF_CASES)
def test_rohf_energy_matches_pyscf(
    name: str,
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    multiplicity: int,
) -> None:
    """Энергия ROHF совпадает с PySCF.

    ROHF отличается от UHF построением плотности (общие орбитали каналов),
    поэтому сверка энергии подтверждает именно этот фокиан, а не переиспользует
    проверку UHF.
    """
    molecule = _rohf_molecule(name, atoms, charge, multiplicity)
    ours = run_rohf(build_basis("sto-3g", molecule), molecule, TIGHT)
    assert ours.converged

    mol = pyscf.gto.M(
        atom=[[symbol, list(position)] for symbol, position in atoms],
        basis="sto-3g",
        spin=multiplicity - 1,
        charge=charge,
        cart=True,
        verbose=0,
    )
    reference = pyscf.scf.ROHF(mol).run(conv_tol=1e-12)
    assert ours.total_energy == pytest.approx(reference.e_tot, abs=1e-7)


@pytest.mark.parametrize(("name", "atoms", "charge", "multiplicity"), ROHF_CASES)
def test_rohf_nuclear_gradient_matches_pyscf(
    name: str,
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    multiplicity: int,
) -> None:
    """Ядерный градиент ROHF против независимого оракула.

    Самая чувствительная проверка: член релаксации строится из фокиан-зависимых
    весов (замкнутый блок по ``Fc``, открытый по ``F^α``, плюс блок связности).
    Блок связности заметен именно на NO, поэтому он обязателен в выборке.
    """
    molecule = _rohf_molecule(name, atoms, charge, multiplicity)
    basis = build_basis("sto-3g", molecule)
    result = run_rohf(basis, molecule, TIGHT)
    ours_gradient = rohf_gradient(basis, molecule, result).gradient

    mol = pyscf.gto.M(
        atom=[[symbol, list(position)] for symbol, position in atoms],
        basis="sto-3g",
        spin=multiplicity - 1,
        charge=charge,
        cart=True,
        verbose=0,
    )
    reference = pyscf.scf.ROHF(mol).run(conv_tol=1e-12)
    theirs_gradient = np.asarray(reference.nuc_grad_method().kernel())

    deviation = float(np.max(np.abs(ours_gradient - theirs_gradient)))
    assert deviation < 1e-6, deviation
