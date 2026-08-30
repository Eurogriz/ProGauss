"""Генерация данных базисных наборов из Basis Set Exchange.

Данные базисов **не пишутся от руки**: они извлекаются из Basis Set Exchange
(авторитетная курируемая библиотека) и сохраняются в репозиторий вместе с
источником. Это исключает риск опечатки в экспонентах и даёт проверяемое
происхождение каждого числа (§26, §54 ТЗ).

Запуск (требуется dev-зависимость ``basis_set_exchange``)::

    python tools/generate_basis_data.py

Повторный запуск идемпотентен: при той же версии BSE файлы не меняются.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import basis_set_exchange as bse

#: Элементы, для которых храним данные (основная органика + третий период).
ELEMENTS: tuple[int, ...] = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17)

#: Наши имена → имена в Basis Set Exchange.
BASIS_SETS: tuple[tuple[str, str], ...] = (
    ("sto-3g", "STO-3G"),
    ("3-21g", "3-21G"),
    ("6-31g", "6-31G"),
    ("6-31g(d)", "6-31G*"),
    ("6-31g(d,p)", "6-31G**"),
    ("6-311g", "6-311G"),
    ("6-311g(d,p)", "6-311G**"),
    ("cc-pvdz", "cc-pVDZ"),
    ("cc-pvtz", "cc-pVTZ"),
    ("cc-pvqz", "cc-pVQZ"),
    ("aug-cc-pvdz", "aug-cc-pVDZ"),
    ("aug-cc-pvtz", "aug-cc-pVTZ"),
    ("def2-svp", "def2-SVP"),
    ("def2-tzvp", "def2-TZVP"),
    ("def2-tzvpp", "def2-TZVPP"),
    ("def2-qzvp", "def2-QZVP"),
)

TARGET = Path(__file__).resolve().parents[1] / "src" / "quantumlab" / "engine" / "basis_data"


#: BSE помечает оболочки тремя способами:
#:   ``gto``            — оболочка без явного указания угловой схемы;
#:   ``gto_cartesian``  — декартовы гауссианы (поляризация в Pople-базисах);
#:   ``gto_spherical``  — чистые угловые моменты (cc-pV*, aug-cc-pV*, def2-*).
#: Радиальная часть (экспоненты и коэффициенты) во всех случаях одинакова,
#: различается только угловая схема. Мы сохраняем все три типа и фиксируем
#: исходный, чтобы движок мог честно сообщить: базис содержит оболочки,
#: определённые в сферической схеме, а расчёт идёт в декартовой.
KNOWN_TYPES: frozenset[str] = frozenset({"gto", "gto_cartesian", "gto_spherical"})


def _shell(raw: dict[str, Any]) -> dict[str, Any]:
    """Преобразует оболочку BSE в компактный внутренний формат."""
    return {
        "angular_momentum": list(raw["angular_momentum"]),
        "exponents": [float(value) for value in raw["exponents"]],
        "coefficients": [[float(value) for value in row] for row in raw["coefficients"]],
        "bse_function_type": raw.get("function_type", "gto"),
    }


def convert(name: str, bse_name: str) -> dict[str, Any]:
    """Скачивает базис из BSE и приводит к внутренней схеме."""
    document = json.loads(bse.get_basis(bse_name, elements=list(ELEMENTS), fmt="json"))
    elements: dict[str, Any] = {}
    for z, payload in document["elements"].items():
        electron_shells = payload.get("electron_shells", [])
        unknown = [
            s.get("function_type")
            for s in electron_shells
            if s.get("function_type") not in KNOWN_TYPES
        ]
        if unknown:
            kinds = sorted(set(unknown))
            msg = f"Базис {bse_name!r}, элемент Z={z}: неизвестный тип оболочки {kinds}"
            raise ValueError(msg)
        shells = [_shell(shell) for shell in electron_shells]
        if not shells:
            continue
        elements[z] = {
            "shells": shells,
            "references": payload.get("references", []),
        }
    has_spherical_shells = any(
        shell["bse_function_type"] == "gto_spherical"
        for entry in elements.values()
        for shell in entry["shells"]
    )
    return {
        "schema_version": "1.0",
        "angular_scheme_published": "spherical" if has_spherical_shells else "cartesian",
        "name": name,
        "display_name": document.get("name", bse_name),
        "bse_name": bse_name,
        "family": document.get("family", ""),
        "description": (document.get("description") or "").strip(),
        "bse_revision_date": document.get("revision_date", ""),
        "source": f"Basis Set Exchange (bse {bse.__version__})",
        "elements": elements,
    }


def main() -> None:
    """Записывает JSON-файлы базисов."""
    TARGET.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, Any]] = {}
    for name, bse_name in BASIS_SETS:
        data = convert(name, bse_name)
        target = TARGET / f"{name}.json"
        text = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        target.write_text(text, encoding="utf-8")
        index[name] = {
            "display_name": data["display_name"],
            "elements": sorted(int(z) for z in data["elements"]),
            "size_bytes": target.stat().st_size,
        }
        size_kb = target.stat().st_size / 1024
        print(f"{name:<14} элементов: {len(data['elements']):>2}  {size_kb:6.1f} КБ")

    total = sum(item["size_bytes"] for item in index.values())
    print(f"всего: {total / 1024:.1f} КБ в {TARGET}")


if __name__ == "__main__":
    main()
