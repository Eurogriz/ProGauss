"""Доменная модель молекулы: состав, заряд/мультиплетность, проверки структуры.

Проверки опираются на известные геометрические эталоны (вода: r(OH) = 0.9578 Å,
угол 104.47°; бензол: C–C = 1.39 Å, C–H = 1.09 Å), поэтому тест проверяет не
«что-то работает», а физически корректное поведение.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from quantumlab.domain.molecule import (
    Atom,
    Bond,
    BondOrder,
    Molecule,
    angle_degrees,
    dihedral_degrees,
    element_from_symbol,
)
from quantumlab.errors import (
    EmptyMoleculeError,
    InvalidMultiplicityError,
    UnknownElementError,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


@pytest.fixture
def benzene() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "benzene.xyz").read_text(encoding="utf-8"), name="benzene")


def test_water_composition(water: Molecule) -> None:
    assert water.formula == "H2O"
    assert water.n_atoms == 3
    assert water.n_electrons == 10
    assert water.n_electrons == 8 + 1 + 1


def test_charge_changes_electron_count(water: Molecule) -> None:
    anion = water.model_copy(update={"charge": -1})
    assert anion.n_electrons == 11


def test_allowed_multiplicities_follow_electron_parity(water: Molecule) -> None:
    # 10 электронов -> неспаренных может быть 0, 2, 4, 6, 8, 10 -> нечётные мультиплетности.
    # Верхняя граница 11 = все электроны неспарены; раньше диапазон обрывался на 9.
    assert water.allowed_multiplicities() == (1, 3, 5, 7, 9, 11)
    anion = water.model_copy(update={"charge": -1})
    assert anion.allowed_multiplicities() == (2, 4, 6, 8, 10, 12)


def test_incompatible_multiplicity_is_rejected_with_suggestion(water: Molecule) -> None:
    with pytest.raises(InvalidMultiplicityError) as info:
        Molecule(name="water", atoms=water.atoms, charge=water.charge, multiplicity=2)
    error = info.value
    assert error.allowed == (1, 3, 5, 7, 9, 11)
    explanation = error.explain("ru")
    assert "Заряд и мультиплетность несовместимы" in error.title("ru")
    assert "Допустимые значения: 1, 3, 5, 7, 9, 11" in explanation


def test_single_electron_system_allows_doublet() -> None:
    """Атомарный водород: 1 электрон -> единственно возможная мультиплетность 2.

    Регрессия: раньше диапазон допустимых значений обрывался на числе электронов,
    список получался пустым и физически корректный дублет отвергался.
    """
    hydrogen = Molecule(
        name="h", atoms=(Atom(symbol="H", position=(0.0, 0.0, 0.0)),), multiplicity=2
    )
    assert hydrogen.allowed_multiplicities() == (2,)
    with pytest.raises(InvalidMultiplicityError) as info:
        Molecule(name="h", atoms=(Atom(symbol="H", position=(0.0, 0.0, 0.0)),))
    assert info.value.allowed == (2,)


def test_empty_molecule_is_rejected() -> None:
    with pytest.raises(EmptyMoleculeError):
        Molecule(name="nothing", atoms=())


def test_unknown_element_is_reported() -> None:
    with pytest.raises(UnknownElementError) as info:
        element_from_symbol("Xx")
    assert info.value.symbol == "Xx"
    assert "Xx" in info.value.explain("ru")


def test_bond_perception_finds_all_benzene_bonds(benzene: Molecule) -> None:
    assert benzene.n_atoms == 12
    assert len(benzene.bonds) == 12
    assert benzene.formula == "C6H6"


def test_water_geometry_is_reproduced_by_measurements(water: Molecule) -> None:
    o, h1, h2 = (atom.position for atom in water.atoms)
    assert math.isclose(math.dist(o, h1), 0.9578, abs_tol=1e-4)
    assert math.isclose(angle_degrees(h1, o, h2), 104.47, abs_tol=0.01)


def test_dihedral_measurement_matches_construction() -> None:
    a = (0.0, 1.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)
    d = (1.0, math.cos(math.radians(60.0)), math.sin(math.radians(60.0)))
    assert math.isclose(abs(dihedral_degrees(a, b, c, d)), 60.0, abs_tol=1e-9)


def test_aromatic_bonds_give_correct_valence(benzene: Molecule) -> None:
    aromatic = tuple(Bond(i=i, j=(i + 1) % 6, order=BondOrder.AROMATIC) for i in range(6)) + tuple(
        Bond(i=i, j=i + 6, order=BondOrder.SINGLE) for i in range(6)
    )
    rebuilt = Molecule(name=benzene.name, atoms=benzene.atoms, bonds=aromatic)
    assert rebuilt.check_valence() == ()
    assert len(rebuilt.bonds) == 12


def test_single_bonds_in_benzene_are_flagged_as_valence_problem(benzene: Molecule) -> None:
    single = tuple(Bond(i=i, j=(i + 1) % 6) for i in range(6)) + tuple(
        Bond(i=i, j=i + 6) for i in range(6)
    )
    molecule = Molecule(name="benzene", atoms=benzene.atoms, bonds=single)
    issues = molecule.check_valence()
    carbons = [issue for issue in issues if issue.symbol == "C"]
    assert len(carbons) == 6
    assert all(issue.observed == 3.0 and issue.expected == 4 for issue in carbons)


def test_suspicious_contact_is_detected() -> None:
    # OH — радикал с 9 электронами, поэтому мультиплетность обязана быть 2
    molecule = Molecule.from_atoms(
        ["O", "H"],
        [(0.0, 0.0, 0.0), (0.30, 0.0, 0.0)],
        multiplicity=2,
        name="too-close",
    )
    contacts = molecule.suspicious_contacts()
    assert len(contacts) == 1
    index_i, index_j, distance = contacts[0]
    assert (index_i, index_j) == (0, 1)
    assert distance < 0.4


def test_xyz_round_trip_preserves_structure(water: Molecule) -> None:
    restored = Molecule.from_xyz(water.to_xyz(), name="water")
    assert restored.atoms == water.atoms
    assert restored.structure_hash() == water.structure_hash()


def test_structure_hash_is_sensitive_to_geometry(water: Molecule) -> None:
    moved = Molecule(
        name="water",
        atoms=(water.atoms[0], water.atoms[1], Atom(symbol="H", position=(0.0, 0.0, 1.0))),
    )
    assert moved.structure_hash() != water.structure_hash()


def test_bond_to_itself_is_rejected() -> None:
    with pytest.raises(ValueError, match="сам с собой"):
        Bond(i=1, j=1)


def test_duplicate_bonds_are_rejected(water: Molecule) -> None:
    with pytest.raises(ValueError, match="дублирующиеся"):
        Molecule(
            name="water",
            atoms=water.atoms,
            bonds=(Bond(i=0, j=1), Bond(i=1, j=0)),
        )
