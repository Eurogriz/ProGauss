"""Пользовательские базисные наборы: разбор файла и каталог пользователя.

Пользовательский базис — это JSON той же схемы, что и данные Basis Set
Exchange в ``basis_data/`` (поэтому остальной код его не отличает), лежащий в
каталоге пользователя. Каталог — общий для всех процессов (CLI, воркер,
``job retry``): имя базиса в спецификации задания остаётся единственным, что
нужно, чтобы воспроизвести расчёт в другом процессе.

Принимаются два формата входного файла:

* **Gaussian94** (``.gbs``, ``.gbs``-подобный текст NWChem/Gaussian из BSE):
  блоки ``Символ 0`` … ``****`` с оболочками ``S|P|D|F|G|H|I|SP  n  scale``;
* **JSON схемы QuantumLab/BSE** — проверяется и копируется как есть.

Каталог: переменная окружения ``QUANTUMLAB_BASIS_DIR`` или ``~/.quantumlab/basis``.
Имя файла (без расширения, в нижнем регистре) — имя базиса; совпадение с
именем встроенного базиса отклоняется, чтобы расчёт под известным именем не
менял смысл молча (§54 ТЗ).

Не поддерживается и отклоняется явно: ECP/псевдопотенциалы, оболочки с
нулевым числом примитивов, неизвестные элементы.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from quantumlab.domain.molecule import ELEMENTS_BY_SYMBOL

#: Переменная окружения с каталогом пользовательских базисов.
BASIS_DIR_ENV = "QUANTUMLAB_BASIS_DIR"

_MOMENTA = "SPDFGHIKLM"
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.()+*,\-]*$")


class CustomBasisError(ValueError):
    """Файл базиса не удалось разобрать или он не проходит проверку."""


def basis_directory() -> Path:
    """Каталог пользовательских базисов (может не существовать)."""
    override = os.environ.get(BASIS_DIR_ENV)
    return Path(override) if override else Path.home() / ".quantumlab" / "basis"


def _to_float(token: str, line_number: int) -> float:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        msg = f"строка {line_number}: «{token}» — не число"
        raise CustomBasisError(msg) from error


def parse_gaussian94(text: str, *, name: str, scheme: str = "spherical") -> dict[str, Any]:
    """Разбирает Gaussian94-текст в сырой словарь схемы ``basis_data``.

    Args:
        text: содержимое файла.
        name: имя базиса (нижний регистр).
        scheme: ``spherical`` или ``cartesian`` — схема, в которой базис
            определён. В формате она не записана, поэтому задаётся явно.

    Raises:
        CustomBasisError: синтаксическая ошибка, ECP, неизвестный элемент.
    """
    if scheme not in ("spherical", "cartesian"):
        msg = f"неизвестная угловая схема {scheme!r}: допустимы spherical и cartesian"
        raise CustomBasisError(msg)
    elements: dict[str, Any] = {}
    lines = [(number, raw.split("!")[0].strip()) for number, raw in enumerate(text.splitlines(), 1)]
    lines = [(number, line) for number, line in lines if line]
    index = 0
    while index < len(lines):
        number, line = lines[index]
        if line.startswith("****"):
            index += 1
            continue
        if line.upper() in ("SPHERICAL", "CARTESIAN"):
            index += 1
            continue
        parts = line.split()
        symbol = parts[0].capitalize()
        if symbol not in ELEMENTS_BY_SYMBOL or len(parts) != 2 or parts[1] != "0":
            msg = f"строка {number}: ожидался заголовок элемента «Символ 0», получено «{line}»"
            raise CustomBasisError(msg)
        element = ELEMENTS_BY_SYMBOL[symbol]
        index += 1
        shells: list[dict[str, Any]] = []
        while index < len(lines) and not lines[index][1].startswith("****"):
            number, line = lines[index]
            header = line.split()
            kind = header[0].upper()
            if kind.endswith("ECP") or "-ECP" in kind or kind == "ECP":
                msg = f"строка {number}: ECP не поддерживаются"
                raise CustomBasisError(msg)
            if len(header) != 3 or not (kind == "SP" or (len(kind) == 1 and kind in _MOMENTA)):
                msg = f"строка {number}: ожидалась оболочка «L n scale», получено «{line}»"
                raise CustomBasisError(msg)
            try:
                n_primitives = int(header[1])
            except ValueError as error:
                msg = f"строка {number}: число примитивов «{header[1]}» не целое"
                raise CustomBasisError(msg) from error
            if n_primitives < 1:
                msg = f"строка {number}: оболочка без примитивов"
                raise CustomBasisError(msg)
            momenta = [0, 1] if kind == "SP" else [_MOMENTA.index(kind)]
            index += 1
            exponents: list[float] = []
            columns: list[list[float]] = [[] for _ in momenta]
            for _ in range(n_primitives):
                if index >= len(lines):
                    msg = f"строка {number}: оболочка оборвана на середине"
                    raise CustomBasisError(msg)
                row_number, row = lines[index]
                tokens = row.split()
                if len(tokens) != 1 + len(momenta):
                    msg = (
                        f"строка {row_number}: ожидалось {1 + len(momenta)} чисел, "
                        f"получено {len(tokens)}"
                    )
                    raise CustomBasisError(msg)
                exponents.append(_to_float(tokens[0], row_number))
                for column, token in zip(columns, tokens[1:], strict=True):
                    column.append(_to_float(token, row_number))
                index += 1
            if any(value <= 0.0 for value in exponents):
                msg = f"строка {number}: экспоненты должны быть положительными"
                raise CustomBasisError(msg)
            shells.append(
                {
                    "angular_momentum": momenta,
                    "bse_function_type": "gto",
                    "exponents": exponents,
                    "coefficients": columns,
                }
            )
        if not shells:
            msg = f"элемент {symbol}: нет ни одной оболочки"
            raise CustomBasisError(msg)
        elements[str(element.z)] = {"shells": shells}
        index += 1  # строка ****
    if not elements:
        msg = "файл не содержит ни одного элемента"
        raise CustomBasisError(msg)
    return _wrap(name, elements, scheme)


def _wrap(name: str, elements: dict[str, Any], scheme: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "name": name,
        "display_name": name,
        "family": "custom",
        "angular_scheme_published": scheme,
        "source": "custom (импортирован пользователем)",
        "elements": elements,
    }


def validate_raw(raw: dict[str, Any], *, name: str) -> dict[str, Any]:
    """Проверяет JSON схемы ``basis_data`` и возвращает нормализованную копию."""
    elements = raw.get("elements")
    if not isinstance(elements, dict) or not elements:
        msg = "JSON базиса: нет непустого поля «elements»"
        raise CustomBasisError(msg)
    scheme = raw.get("angular_scheme_published", "spherical")
    result = _wrap(name, {}, str(scheme))
    if scheme not in ("spherical", "cartesian"):
        msg = f"неизвестная угловая схема {scheme!r}"
        raise CustomBasisError(msg)
    for key, entry in elements.items():
        if not str(key).isdigit() or not isinstance(entry, dict) or "shells" not in entry:
            msg = f"JSON базиса: элемент {key!r} должен быть вида {{Z: {{shells: [...]}}}}"
            raise CustomBasisError(msg)
        for shell in entry["shells"]:
            exponents = shell.get("exponents")
            coefficients = shell.get("coefficients")
            momenta = shell.get("angular_momentum")
            if not exponents or not coefficients or not momenta:
                msg = f"JSON базиса: неполная оболочка у элемента {key}"
                raise CustomBasisError(msg)
            if any(len(row) != len(exponents) for row in coefficients):
                msg = f"JSON базиса: число коэффициентов не равно числу экспонент у элемента {key}"
                raise CustomBasisError(msg)
            if any(float(value) <= 0.0 for value in exponents):
                msg = f"JSON базиса: неположительная экспонента у элемента {key}"
                raise CustomBasisError(msg)
        result["elements"][str(int(key))] = entry
    return result


def normalize_name(name: str) -> str:
    """Имя базиса в каноническом виде; недопустимое — ошибка."""
    normalized = name.strip().lower()
    if not _NAME_PATTERN.match(normalized):
        msg = f"недопустимое имя базиса {name!r}: латиница, цифры и _.()+*,-"
        raise CustomBasisError(msg)
    return normalized


def import_basis_file(
    source: Path,
    *,
    name: str | None = None,
    scheme: str = "spherical",
    directory: Path | None = None,
    builtin_names: frozenset[str] = frozenset(),
) -> Path:
    """Разбирает файл и сохраняет его в каталоге пользовательских базисов.

    Args:
        source: Gaussian94 (любое расширение, кроме ``.json``) или JSON.
        name: имя базиса; по умолчанию — имя файла без расширения.
        scheme: угловая схема для Gaussian94-файла (JSON несёт её сам).
        directory: каталог назначения; по умолчанию :func:`basis_directory`.
        builtin_names: имена встроенных базисов — перезаписать их нельзя.

    Returns:
        Путь к сохранённому JSON.
    """
    resolved = normalize_name(name or source.stem)
    if resolved in builtin_names:
        msg = f"имя {resolved!r} занято встроенным базисом; выберите другое (--name)"
        raise CustomBasisError(msg)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            msg = f"{source}: некорректный JSON ({error})"
            raise CustomBasisError(msg) from error
        raw = validate_raw(payload, name=resolved)
    else:
        raw = parse_gaussian94(text, name=resolved, scheme=scheme)
    target_dir = directory or basis_directory()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{resolved}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(raw, indent=1, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def custom_basis_names() -> tuple[str, ...]:
    """Имена базисов из каталога пользователя."""
    directory = basis_directory()
    if not directory.is_dir():
        return ()
    return tuple(sorted(path.stem for path in directory.glob("*.json")))


def load_custom_raw(name: str) -> dict[str, Any] | None:
    """Читает пользовательский базис; ``None``, если такого нет.

    Содержимое хешируется и попадает в ``display_name``: два файла под одним
    именем — это разные расчёты, и отпечаток результата должен это показывать.
    """
    try:
        normalized = normalize_name(name)
    except CustomBasisError:
        return None
    path = basis_directory() / f"{normalized}.json"
    if not path.is_file():
        return None
    payload = path.read_bytes()
    raw: dict[str, Any] = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()[:8]
    raw["display_name"] = f"{raw.get('display_name', normalized)} (custom {digest})"
    raw["name"] = normalized
    return raw
