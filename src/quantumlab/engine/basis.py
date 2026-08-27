"""Библиотека базисных наборов.

Данные читаются из JSON-файлов в ``basis_data/``, сгенерированных из Basis Set
Exchange (``tools/generate_basis_data.py``). Каждое число имеет происхождение;
руками экспоненты не правятся.

Соглашения:

* **декартовы гауссианы**: для ``l = 2`` это 6 функций (xx, yy, zz, xy, xz, yz).
  Выбор осознанный: декартова схема не требует матриц перехода к сферическим
  гармоникам и однозначна при сравнении с внешними пакетами (у них нужно
  включать ``cart=True``);
* каждая сжатая оболочка нормируется на единицу — на энергию это не влияет
  (линейная перепараметризация базиса), но улучшает обусловленность матриц;
* порядок компонент внутри оболочки фиксирован и задан функцией
  :func:`cartesian_powers`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, cast

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.constants import angstrom_to_bohr

# `as` — явный ре-экспорт: mypy --strict запрещает неявный (no_implicit_reexport).
from quantumlab.errors import BasisNotFoundError as BasisNotFoundError


class BasisDataError(RuntimeError):
    """Данные базиса повреждены или не содержат нужного элемента."""


@dataclass(frozen=True, slots=True)
class Shell:
    """Сжатая оболочка: центр, момент, экспоненты и коэффициенты.

    Attributes:
        center: индекс атома в молекуле.
        angular_momentum: квантовое число ``l``.
        exponents: экспоненты примитивов (а.е.).
        coefficients: коэффициенты сжатия, уже умноженные на нормы примитивов.
        n_primitives: число примитивов.
    """

    center: int
    angular_momentum: int
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]

    @property
    def n_primitives(self) -> int:
        """Число примитивных гауссиан."""
        return len(self.exponents)

    @property
    def n_cartesian(self) -> int:
        """Число декартовых компонент для данного ``l``."""
        l_value = self.angular_momentum
        return (l_value + 1) * (l_value + 2) // 2

    @property
    def component_scales(self) -> tuple[float, ...]:
        """Поправки к норме для каждой декартовой компоненты.

        Коэффициенты оболочки нормированы по канонической компоненте ``(l,0,0)``.
        Чтобы каждая декартова функция имела единичную норму, интегральный
        движок умножает коэффициент на этот множитель: для ``s`` и ``p`` он
        равен 1, для ``d`` — 1 у ``xx, yy, zz`` и ``√3`` у ``xy, xz, yz``.
        """
        return tuple(
            _component_scale(self.angular_momentum, powers)
            for powers in cartesian_powers(self.angular_momentum)
        )


def cartesian_powers(l_value: int) -> tuple[tuple[int, int, int], ...]:
    """Декартовы степени ``(lx, ly, lz)`` для момента ``l``.

    Порядок фиксирован и используется везде: при сборке матриц и при выдаче
    орбиталей, поэтому его нельзя менять без пересчёта сохранённых артефактов.
    """
    return tuple(
        (lx, l_value - lx - lz, lz)
        for lx in range(l_value, -1, -1)
        for lz in range(l_value - lx + 1)
    )


def _double_factorial(value: int) -> int:
    result = 1
    while value > 1:
        result *= value
        value -= 2
    return result


def _primitive_norm(exponent: float, l_value: int) -> float:
    r"""Норма примитива для «канонической» компоненты оболочки ``(l, 0, 0)``.

    Полная норма декартова гауссиана ``x^{lx} y^{ly} z^{lz} e^{-αr²}`` равна

    .. math::

        N = \\left[ (2α/π)^{3/2} (4α)^l \big/ \\prod_i (2l_i - 1)!!
    ight]^{1/2},

    то есть в знаменателе стоит **произведение двойных факториалов по осям**,
    а не ``(2l−1)!!`` для полного ``l``. Для компонент вроде ``xy`` эти
    величины различаются (``1!!·1!! = 1`` против ``3!! = 3``), поэтому
    коэффициенты оболочки нормируются по канонической компоненте, а
    покомпонентную поправку возвращает :meth:`Shell.component_scales`.
    """
    return math.sqrt(
        (2.0 * exponent / math.pi) ** 1.5
        * (4.0 * exponent) ** l_value
        / _double_factorial(2 * l_value - 1)
    )


def _component_scale(l_value: int, powers: tuple[int, int, int]) -> float:
    """Поправка к норме для декартовой компоненты ``powers`` оболочки ``l``.

    Равна ``√((2l−1)!! / ∏_i (2l_i−1)!!)``. Для ``s`` и ``p`` всегда 1, для
    ``d``: 1 у ``xx, yy, zz`` и ``√3`` у ``xy, xz, yz``.
    """
    total = _double_factorial(2 * l_value - 1)
    per_axis = 1
    for power in powers:
        per_axis *= _double_factorial(2 * power - 1)
    return math.sqrt(total / per_axis)


@dataclass(frozen=True, slots=True)
class BasisSet:
    """Базисный набор, развёрнутый по атомам конкретной молекулы."""

    name: str
    display_name: str
    shells: tuple[Shell, ...]

    @property
    def n_functions(self) -> int:
        """Полное число базисных функций."""
        return sum(shell.n_cartesian for shell in self.shells)

    def shell_slices(self) -> Iterator[tuple[int, int, int]]:
        """Итератор ``(номер оболочки, начало среза, конец среза)``."""
        offset = 0
        for index, shell in enumerate(self.shells):
            yield index, offset, offset + shell.n_cartesian
            offset += shell.n_cartesian


@cache
def _load_raw(name: str) -> dict[str, Any]:
    """Читает JSON базиса по имени."""
    normalized = name.strip().lower()
    resource = resources.files("quantumlab.engine.basis_data")
    available = {
        path.name[:-5] for path in cast("Any", resource.iterdir()) if path.name.endswith(".json")
    }
    if normalized not in available:
        raise BasisNotFoundError(name)
    payload: Any = json.loads(resource.joinpath(f"{normalized}.json").read_text(encoding="utf-8"))
    return cast("dict[str, Any]", payload)


def available_basis_sets() -> tuple[str, ...]:
    """Имена базисов, для которых есть данные."""
    resource = resources.files("quantumlab.engine.basis_data")
    names = [
        path.name[:-5] for path in cast("Any", resource.iterdir()) if path.name.endswith(".json")
    ]
    return tuple(sorted(names))


def build_basis(name: str, molecule: Molecule) -> BasisSet:
    """Разворачивает базисный набор по атомам молекулы.

    Каждая сжатая оболочка нормируется: коэффициенты умножаются на нормы
    примитивов и на общий множитель, приводящий норму сжатой функции к 1.
    """
    raw = _load_raw(name)
    shells: list[Shell] = []

    for atom_index, atom in enumerate(molecule.atoms):
        entry = raw["elements"].get(str(atom.z))
        if entry is None:
            msg = (
                f"Базисный набор {name!r} не содержит данных для элемента "
                f"{atom.symbol} (Z={atom.z}). Выберите другой базис или добавьте данные."
            )
            raise BasisDataError(msg)
        for shell_data in entry["shells"]:
            exponents = tuple(shell_data["exponents"])
            for l_value, raw_coefficients in _contractions(shell_data):
                primitive_norms = tuple(
                    _primitive_norm(exponent, l_value) for exponent in exponents
                )
                norm = _contracted_norm(exponents, raw_coefficients, primitive_norms, l_value)
                shells.append(
                    Shell(
                        center=atom_index,
                        angular_momentum=l_value,
                        exponents=exponents,
                        coefficients=tuple(
                            c * n / norm
                            for c, n in zip(raw_coefficients, primitive_norms, strict=True)
                        ),
                    )
                )

    return BasisSet(name=raw["name"], display_name=raw["display_name"], shells=tuple(shells))


def _contractions(shell_data: dict[str, Any]) -> list[tuple[int, tuple[float, ...]]]:
    """Разворачивает оболочку BSE в список ``(l, коэффициенты)`` сжатых функций.

    В схеме BSE **число строк матрицы коэффициентов равно числу сжатых
    функций**, а ``angular_momentum`` перечисляет только различные значения
    ``l``. Встречаются ровно две формы:

    * ``len(coefficients) == len(angular_momentum)`` — строки соответствуют
      значениям ``l`` по порядку (так устроены общие SP-оболочки базисов
      Pople: строка 0 — это s, строка 1 — это p);
    * ``len(angular_momentum) == 1`` при любом числе строк — все строки
      относятся к одному и тому же ``l`` (так хранятся cc-pV*, aug-cc-pV*:
      одна оболочка несёт сразу 1s, 2s и 3s).

    Иное сочетание означает неизвестную форму; данные не интерпретируются,
    чтобы не выдать усечённый базис за полный.
    """
    momenta: list[int] = list(shell_data["angular_momentum"])
    rows = [tuple(float(value) for value in row) for row in shell_data["coefficients"]]
    if len(rows) == len(momenta):
        return list(zip(momenta, rows, strict=True))
    if len(momenta) == 1:
        return [(momenta[0], row) for row in rows]
    msg = (
        "Неизвестная форма оболочки базиса: "
        f"{len(momenta)} значений углового момента и {len(rows)} строк коэффициентов. "
        "Правило сопоставления неоднозначно, оболочка не интерпретируется."
    )
    raise BasisDataError(msg)


def _contracted_norm(
    exponents: tuple[float, ...],
    coefficients: tuple[float, ...],
    primitive_norms: tuple[float, ...],
    l_value: int,
) -> float:
    """Норма сжатой функции.

    Норма в квадрате — ``Σ_ij c_i c_j N_i N_j ⟨g_i|g_j⟩``, где ``c`` — исходные
    коэффициенты из файла базиса, ``N`` — нормы примитивов, а ``⟨g_i|g_j⟩`` —
    перекрывание **ненормированных** примитивов. Смешивать нормированные и
    ненормированные примитивы в этом выражении нельзя: двойной учёт норм ломает
    масштаб базисных функций.
    """
    total = 0.0
    for i, (alpha_i, c_i) in enumerate(zip(exponents, coefficients, strict=True)):
        for j, (alpha_j, c_j) in enumerate(zip(exponents, coefficients, strict=True)):
            total += (
                c_i
                * c_j
                * primitive_norms[i]
                * primitive_norms[j]
                * _same_angular_overlap(alpha_i, alpha_j, l_value)
            )
    return math.sqrt(total)


def _same_angular_overlap(alpha: float, beta: float, l_value: int) -> float:
    """Перекрывание двух нормированных примитивов с одинаковым ``l``."""
    return float(
        _double_factorial(2 * l_value - 1)
        / (2.0 * (alpha + beta)) ** l_value
        * (math.pi / (alpha + beta)) ** 1.5
    )


@cache
def basis_angular_scheme(name: str) -> str:
    """Угловая схема, в которой базис опубликован: ``cartesian`` или ``spherical``.

    Поле записывается генератором данных по фактическим меткам Basis Set
    Exchange. Оно нужно для честности: наш движок считает в декартовой схеме,
    и для базисов, опубликованных в сферической, это **больший** базис
    (6 d-функций вместо 5). Расхождение с табличными энергиями ~1e-4 Eh
    движок обязан показывать, а не прятать.
    """
    raw = _load_raw(name)
    scheme = raw.get("angular_scheme_published")
    return "spherical" if scheme == "spherical" else "cartesian"


def nuclear_repulsion(molecule: Molecule) -> float:
    """Энергия отталкивания ядер в атомных единицах."""
    total = 0.0
    atoms = molecule.atoms
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = math.dist(
                [angstrom_to_bohr(v) for v in atoms[i].position],
                [angstrom_to_bohr(v) for v in atoms[j].position],
            )
            total += atoms[i].z * atoms[j].z / distance
    return total
