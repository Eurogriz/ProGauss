"""Модель молекулы: атомы, связи, заряд/мультиплетность, проверка структуры.

Единицы (§23 ТЗ): координаты — ангстремы (Å), заряд — элементарные заряды.
Энергии во всём пакете — хартри (Eh), частоты — см⁻¹.

Модель намеренно **физически проверяема**: число электронов, допустимая
мультиплетность и валентности вычисляются из состава, а не хранятся как
данные, которые пользователь мог бы рассогласовать.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quantumlab.errors import EmptyMoleculeError, InvalidMultiplicityError, UnknownElementError

# --------------------------------------------------------------------------- #
# Справочник элементов.
#
# Ковалентные радиусы — Cordero et al., Dalton Trans., 2008, 2832
# (https://doi.org/10.1039/B801115J). Радиусы используются ТОЛЬКО для
# геометрических проверок (восприятие связей, поиск подозрительных контактов)
# и не входят ни в один физический расчёт.
#
# ``typical_valence`` — типичная валентность для нейтральных органических
# соединений. Для переходных металлов она не определена (None), поэтому
# проверка валентности для них честно пропускается, а не «угадывается».
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Element:
    """Химический элемент."""

    z: int
    symbol: str
    name_ru: str
    covalent_radius: float
    typical_valence: int | None
    mass_amu: float


#: Стандартные атомные массы (IUPAC). Последний аргумент — ``mass_amu``: массы
#: нужны для масс-взвешенного гессиана в колебательной задаче, без них частоты
#: не посчитать. Значения сверяются тестом с независимой таблицей.
ELEMENTS: tuple[Element, ...] = (
    Element(1, "H", "Водород", 0.31, 1, 1.008),
    Element(5, "B", "Бор", 0.84, 3, 10.81),
    Element(6, "C", "Углерод", 0.76, 4, 12.011),
    Element(7, "N", "Азот", 0.71, 3, 14.007),
    Element(8, "O", "Кислород", 0.66, 2, 15.999),
    Element(9, "F", "Фтор", 0.57, 1, 18.998403163),
    Element(14, "Si", "Кремний", 1.11, 4, 28.085),
    Element(15, "P", "Фосфор", 1.07, 3, 30.973761998),
    Element(16, "S", "Сера", 1.05, 2, 32.06),
    Element(17, "Cl", "Хлор", 1.02, 1, 35.45),
    Element(35, "Br", "Бром", 1.20, 1, 79.904),
    Element(53, "I", "Иод", 1.39, 1, 126.90447),
)

ELEMENTS_BY_SYMBOL: dict[str, Element] = {element.symbol: element for element in ELEMENTS}
ELEMENTS_BY_Z: dict[int, Element] = {element.z: element for element in ELEMENTS}


def element_from_symbol(symbol: str) -> Element:
    """Возвращает элемент по символу; неизвестный символ — :class:`UnknownElementError`."""
    normalized = symbol.strip().capitalize() if len(symbol.strip()) > 1 else symbol.strip().upper()
    element = ELEMENTS_BY_SYMBOL.get(normalized)
    if element is None:
        raise UnknownElementError(symbol)
    return element


class BondOrder(StrEnum):
    """Порядок химической связи.

    Ароматическая связь при подсчёте валентности учитывается как 1.5 —
    это позволяет корректно проверить, например, углерод в бензоле (2×1.5 + 1 = 4).
    """

    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    AROMATIC = "aromatic"

    @property
    def valence_contribution(self) -> float:
        """Вклад связи в валентность атома."""
        return {
            BondOrder.SINGLE: 1.0,
            BondOrder.DOUBLE: 2.0,
            BondOrder.TRIPLE: 3.0,
            BondOrder.AROMATIC: 1.5,
        }[self]


class Atom(BaseModel):
    """Атом: элемент, декартовы координаты в ангстремах, необязательная метка."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(description="Символ элемента, например 'C'")
    position: tuple[float, float, float] = Field(description="Координаты в Å")
    label: str | None = Field(default=None, description="Пользовательская метка атома")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        element = element_from_symbol(value)
        return element.symbol

    @field_validator("position")
    @classmethod
    def _validate_position(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(math.isnan(component) or math.isinf(component) for component in value):
            msg = "Координаты атома должны быть конечными числами"
            raise ValueError(msg)
        return value

    @property
    def z(self) -> int:
        """Атомный номер."""
        return element_from_symbol(self.symbol).z

    @property
    def element(self) -> Element:
        """Справочная запись элемента."""
        return element_from_symbol(self.symbol)


class Bond(BaseModel):
    """Связь между двумя атомами (индексы — позиции в :attr:`Molecule.atoms`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    i: int = Field(ge=0)
    j: int = Field(ge=0)
    order: BondOrder = BondOrder.SINGLE

    @model_validator(mode="after")
    def _validate_pair(self) -> Self:
        if self.i == self.j:
            msg = "Связь не может соединять атом сам с собой"
            raise ValueError(msg)
        return self

    @property
    def pair(self) -> tuple[int, int]:
        """Каноническая (упорядоченная) пара индексов."""
        return (self.i, self.j) if self.i < self.j else (self.j, self.i)


@dataclass(frozen=True, slots=True)
class ValenceIssue:
    """Нарушение валентности, найденное при проверке структуры (§4 ТЗ)."""

    index: int
    symbol: str
    observed: float
    expected: int


class Molecule(BaseModel):
    """Молекула: состав, геометрия, заряд, мультиплетность, связи.

    Инварианты проверяются при создании: мультиплетность обязана быть
    согласована с числом электронов, вычисляемым из состава и заряда.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "molecule"
    atoms: tuple[Atom, ...] = ()
    bonds: tuple[Bond, ...] = ()
    charge: int = Field(default=0, description="Суммарный заряд в элементарных зарядах")
    multiplicity: int = Field(default=1, ge=1, description="Спиновая мультиплетность 2S+1")

    # -- базовые производные величины --------------------------------------- #
    @property
    def n_atoms(self) -> int:
        """Число атомов."""
        return len(self.atoms)

    @property
    def n_electrons(self) -> int:
        """Число электронов: сумма атомных номеров минус заряд."""
        return sum(atom.z for atom in self.atoms) - self.charge

    @property
    def formula(self) -> str:
        """Брутто-формула в нотации Хилла (C и H первыми, остальные по алфавиту)."""
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.symbol] = counts.get(atom.symbol, 0) + 1

        def render(symbol: str) -> str:
            count = counts[symbol]
            return symbol if count == 1 else f"{symbol}{count}"

        head = [render(symbol) for symbol in ("C", "H") if symbol in counts]
        tail = [render(symbol) for symbol in sorted(counts) if symbol not in ("C", "H")]
        return "".join(head + tail) if (head or tail) else ""

    # -- проверки ------------------------------------------------------------ #
    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        if not self.atoms:
            raise EmptyMoleculeError
        n_atoms = len(self.atoms)
        for bond in self.bonds:
            if bond.i >= n_atoms or bond.j >= n_atoms:
                msg = f"Связь {bond.i}-{bond.j} ссылается на несуществующий атом (всего {n_atoms})"
                raise ValueError(msg)
        duplicates = len({bond.pair for bond in self.bonds})
        if duplicates != len(self.bonds):
            msg = "Обнаружены дублирующиеся связи"
            raise ValueError(msg)
        self._validate_multiplicity()
        return self

    def with_state(self, *, charge: int, multiplicity: int) -> Molecule:
        """Возвращает копию с другим зарядом и мультиплетностью.

        Именно копию через валидацию, а не ``model_copy(update=...)``: тот
        обходит валидатор модели, и на выходе получилась бы молекула, у которой
        мультиплетность несовместима с числом электронов. Ошибка всплыла бы
        только внутри расчёта — и не в виде понятной диагностики (§19 ТЗ).
        """
        return Molecule.model_validate(
            {**self.model_dump(), "charge": charge, "multiplicity": multiplicity}
        )

    def allowed_multiplicities(self) -> tuple[int, ...]:
        """Все мультиплетности, совместимые с текущим числом электронов.

        Условие: число неспаренных электронов ``m - 1`` не превышает числа
        электронов и имеет ту же чётность, что и оно.
        """
        electrons = self.n_electrons
        if electrons <= 0:
            return (1,)
        # Максимальная мультиплетность — electrons + 1 (все электроны неспарены),
        # поэтому верхняя граница диапазона на единицу больше числа электронов.
        # Граница ``electrons + 1`` давала пустой список для одного электрона и
        # отвергала атомарный водород — физически корректный дублет.
        return tuple(m for m in range(1, electrons + 2) if (electrons - (m - 1)) % 2 == 0)

    def _validate_multiplicity(self) -> None:
        if (
            self.multiplicity - 1 > self.n_electrons
            or self.multiplicity not in self.allowed_multiplicities()
        ):
            raise InvalidMultiplicityError(
                charge=self.charge,
                electrons=self.n_electrons,
                multiplicity=self.multiplicity,
                allowed=self.allowed_multiplicities(),
            )

    def check_valence(self) -> tuple[ValenceIssue, ...]:
        """Проверяет валентности по насчитанным связям.

        Возвращает список нарушений (пустой, если всё в порядке). Атомы без
        определённой типичной валентности (переходные металлы) пропускаются —
        система не выдаёт догадку за проверку.
        """
        contributions: dict[int, float] = dict.fromkeys(range(self.n_atoms), 0.0)
        for bond in self.bonds:
            contributions[bond.i] += bond.order.valence_contribution
            contributions[bond.j] += bond.order.valence_contribution

        issues: list[ValenceIssue] = []
        for index, atom in enumerate(self.atoms):
            expected = atom.element.typical_valence
            if expected is None:
                continue
            observed = contributions[index]
            if round(observed) != expected:
                issues.append(ValenceIssue(index, atom.symbol, observed, expected))
        return tuple(issues)

    def perceive_bonds(self, *, tolerance: float = 1.3, min_distance: float = 0.4) -> Molecule:
        """Восстанавливает связи из геометрии по ковалентным радиусам.

        Алгоритм: связь принимается, если ``r_ij < tolerance * (r_i + r_j)`` и
        ``r_ij > min_distance``. Сложность ``O(N^2)`` — для крупных систем нужен
        cell-list, это осознанный компромисс препроцессора (не расчётного ядра).

        Найденным связям присваивается порядок SINGLE: порядок связи по одной
        только геометрии не определяется, и система не делает вид, что знает его.
        """
        bonds: list[Bond] = []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                distance = _distance(self.atoms[i].position, self.atoms[j].position)
                limit = tolerance * (
                    self.atoms[i].element.covalent_radius + self.atoms[j].element.covalent_radius
                )
                if min_distance < distance < limit:
                    bonds.append(Bond(i=i, j=j, order=BondOrder.SINGLE))
        return self.model_copy(update={"bonds": tuple(bonds)})

    def suspicious_contacts(self, *, tolerance: float = 0.7) -> tuple[tuple[int, int, float], ...]:
        """Находит слишком близкие несвязанные контакты — типичная ошибка импорта."""
        problems: list[tuple[int, int, float]] = []
        bonded = {bond.pair for bond in self.bonds}
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                if (i, j) in bonded:
                    continue
                distance = _distance(self.atoms[i].position, self.atoms[j].position)
                limit = tolerance * (
                    self.atoms[i].element.covalent_radius + self.atoms[j].element.covalent_radius
                )
                if distance < limit:
                    problems.append((i, j, distance))
        return tuple(problems)

    # -- отпечаток ----------------------------------------------------------- #
    def structure_hash(self) -> str:
        """Устойчивый SHA-256 отпечаток структуры (состав + геометрия + состояние).

        Используется для кэширования и для проверки, что рестарт выполняется
        на той же структуре. Координаты округляются до 1e-8 Å, чтобы отпечаток
        не менялся от шума формата файла.
        """
        parts = [self.charge, self.multiplicity]
        payload = "|".join(
            [
                f"{atom.symbol}:" + ",".join(f"{value:.8f}" for value in atom.position)
                for atom in self.atoms
            ]
            + [f"q={parts[0]}", f"m={parts[1]}"]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- ввод/вывод ---------------------------------------------------------- #
    def to_xyz(self, *, comment: str | None = None) -> str:
        """Сериализует структуру в формат XYZ (координаты в Å)."""
        header = (
            comment
            if comment is not None
            else f"{self.name} | {self.formula} | q={self.charge} m={self.multiplicity}"
        )
        lines = [str(self.n_atoms), header]
        lines.extend(
            "{:<2s} {:>18.10f} {:>18.10f} {:>18.10f}".format(atom.symbol, *atom.position)
            for atom in self.atoms
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_xyz(
        cls, text: str, *, name: str = "molecule", charge: int = 0, multiplicity: int = 1
    ) -> Molecule:
        """Разбирает XYZ-файл. Связи восстанавливаются из геометрии.

        Заряд и мультиплетность задаются сразу, а не через ``with_state`` после
        разбора: валидатор модели проверяет их совместимость с числом электронов
        при создании, и у радикала конструктор с мультиплетностью 1 упал бы ещё
        до того, как вызывающая сторона успела бы её исправить.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise EmptyMoleculeError
        try:
            count = int(lines[0])
        except ValueError as exc:
            msg = "Первая строка XYZ-файла должна содержать число атомов"
            raise ValueError(msg) from exc
        body = lines[2 : 2 + count]
        if len(body) != count:
            msg = f"В XYZ-файле заявлено {count} атомов, найдено {len(body)}"
            raise ValueError(msg)

        atoms: list[Atom] = []
        for line in body:
            tokens = line.split()
            if len(tokens) < 4:
                msg = f"Строка XYZ должна иметь вид 'Символ x y z': {line!r}"
                raise ValueError(msg)
            atoms.append(
                Atom(
                    symbol=element_from_symbol(tokens[0]).symbol,
                    position=(float(tokens[1]), float(tokens[2]), float(tokens[3])),
                )
            )
        molecule = cls(name=name, atoms=tuple(atoms), charge=charge, multiplicity=multiplicity)
        return molecule.perceive_bonds()

    @classmethod
    def from_atoms(
        cls,
        species: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        *,
        charge: int = 0,
        multiplicity: int = 1,
        name: str = "molecule",
    ) -> Molecule:
        """Создаёт молекулу из параллельных списков символов и координат."""
        if len(species) != len(coordinates):
            msg = "Число символов и число координатных троек должно совпадать"
            raise ValueError(msg)
        atoms = tuple(
            Atom(symbol=symbol, position=(float(pos[0]), float(pos[1]), float(pos[2])))
            for symbol, pos in zip(species, coordinates, strict=True)
        )
        return cls(name=name, atoms=atoms, charge=charge, multiplicity=multiplicity)

    def model_dump_canonical(self) -> dict[str, Any]:
        """Словарь с устойчивым порядком ключей — основа для отпечатков расчёта."""
        return {
            "atoms": [
                {"symbol": atom.symbol, "position": list(atom.position)} for atom in self.atoms
            ],
            "bonds": [
                {"i": bond.i, "j": bond.j, "order": bond.order.value}
                for bond in sorted(self.bonds, key=lambda b: b.pair)
            ],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def angle_degrees(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    """Угол a-b-c в градусах (инструмент измерения в редакторе, §4 ТЗ)."""
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    n1 = math.sqrt(sum(component * component for component in v1))
    n2 = math.sqrt(sum(component * component for component in v2))
    if n1 == 0.0 or n2 == 0.0:
        msg = "Угол не определён для совпадающих точек"
        raise ValueError(msg)
    cosine = sum(x * y for x, y in zip(v1, v2, strict=True)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def dihedral_degrees(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> float:
    """Двугранный угол a-b-c-d в градусах в диапазоне (-180, 180]."""

    def sub(p: Iterable[float], q: Iterable[float]) -> tuple[float, float, float]:
        px, py, pz = p
        qx, qy, qz = q
        return (px - qx, py - qy, pz - qz)

    def cross(
        p: tuple[float, float, float], q: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return (
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        )

    def dot(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]

    b1 = sub(b, a)
    b2 = sub(c, b)
    b3 = sub(d, c)
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    m1 = cross(
        n1,
        (
            b2[0] / math.sqrt(dot(b2, b2)),
            b2[1] / math.sqrt(dot(b2, b2)),
            b2[2] / math.sqrt(dot(b2, b2)),
        ),
    )
    x = dot(n1, n2)
    y = dot(m1, n2)
    return math.degrees(math.atan2(y, x))
