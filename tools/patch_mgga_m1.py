"""Одноразовый патч: протокол meta-GGA (стадия M1).

Добавляет параметр ``tau`` в ``evaluate`` всех существующих функционалов
(они его игнорируют — ``del tau``), атрибут ``requires_tau: bool = False``
во все классы и обновляет типизацию FUNCTIONALS/get_functional.
Новые классы TPSS/PBC/TPSSh дописываются вручную в тот же модуль.
"""

from __future__ import annotations

import pathlib
import re

p = pathlib.Path("src/quantumlab/engine/functional.py")
src = p.read_text()

# --- 1) сигнатуры evaluate: +tau (только методы, возвращающие XcEvaluation) ---
old_sig = "        spin_polarized: bool = False,\n    ) -> XcEvaluation:"
new_sig = (
    "        spin_polarized: bool = False,\n"
    "        tau: Array | None = None,\n"
    "    ) -> XcEvaluation:"
)
n_sig = src.count(old_sig)
assert n_sig == 12, f"ожидалось 12 сигнатур evaluate, найдено {n_sig}"
src = src.replace(old_sig, new_sig)

# --- 2) del tau сразу после докстринга каждого evaluate (докстринг сохраняется) ---
pattern = re.compile(r"\) -> XcEvaluation:\n")
out: list[str] = []
pos = 0
n_del = 0
for m in pattern.finditer(src):
    out.append(src[pos : m.end()])
    line_end = src.index("\n", m.end()) + 1  # конец строки-начала докстринга
    first_line = src[m.end() : line_end].strip()
    assert first_line.startswith('"""'), first_line
    if len(first_line) > 3 and first_line.endswith('"""'):
        # однострочный докстринг: позиция после замыкающей """ (без \n)
        doc_end = m.end() + len(src[m.end() : line_end].rstrip("\n"))
    else:
        # многострочный: строка-закрытие "\n        """
        doc_end = src.index('\n        """', line_end) + len('\n        """')
    out.append(src[m.end() : doc_end])  # сам докстринг
    out.append("\n        del tau\n")
    pos = doc_end
    n_del += 1
out.append(src[pos:])
src = "".join(out)
assert n_del == 12, f"ожидалось 12 вставок del tau, сделано {n_del}"

# --- 3) requires_tau в property-классах (перед @property def name) ---
anchor_name = "    @property\n    def name(self) -> str:"
n_name = src.count(anchor_name)
assert n_name == 8, f"ожидалось 8 якорей name, найдено {n_name}"
src = src.replace(anchor_name, "    requires_tau: bool = False\n\n" + anchor_name)

# --- 3b) requires_tau в attribute-классах (после exact_exchange_fraction) ---
for val in ("0.0", "0.25", "0.20"):
    anchor = f"    exact_exchange_fraction: float = {val}\n"
    repl = anchor + "    requires_tau: bool = False\n"
    cnt = src.count(anchor)
    assert cnt >= 1, val
    src = src.replace(anchor, repl)

# --- 4) FUNCTIONALS и get_functional: +Tpssh (класс появится после патча) ---
old_fn = (
    "FUNCTIONALS: dict[str, type[Svwn] | type[Pbe] | type[Pbe0] | type[Blyp] | type[B3lyp]] = {\n"
    '    "svwn": Svwn,\n'
    '    "lda": Svwn,\n'
    '    "pbe": Pbe,\n'
    '    "blyp": Blyp,\n'
    '    "pbe0": Pbe0,\n'
    '    "b3lyp": B3lyp,\n'
    "}"
)
new_fn = (
    "FUNCTIONALS: dict[\n"
    "    str, type[Svwn] | type[Pbe] | type[Pbe0] | type[Blyp] | type[B3lyp] | type[Tpssh]\n"
    "] = {\n"
    '    "svwn": Svwn,\n'
    '    "lda": Svwn,\n'
    '    "pbe": Pbe,\n'
    '    "blyp": Blyp,\n'
    '    "pbe0": Pbe0,\n'
    '    "b3lyp": B3lyp,\n'
    '    "tpssh": Tpssh,\n'
    "}"
)
assert old_fn in src
src = src.replace(old_fn, new_fn)

old_gf = "def get_functional(name: str) -> Svwn | Pbe | Pbe0 | Blyp | B3lyp:"
new_gf = "def get_functional(name: str) -> Svwn | Pbe | Pbe0 | Blyp | B3lyp | Tpssh:"
assert old_gf in src
src = src.replace(old_gf, new_gf)

p.write_text(src)
print("OK: сигнатуры=12, del_tau=12, name-якоря=8, exact_fraction якоря добавлены")
