"""Генератор данных модели дисперсионного взаимодействия D4.

Данные **не пишутся от руки**: они извлечены из Fortran-таблиц официальной
реализации dftd4 (Grimme et al., J. Chem. Theory Comput. 2021, 17, 6579;
мctc-lib / multicharge / dftd4) и зафиксированы в
``src/quantumlab/engine/dispersion_d4_data.py``. Библиотека libdftd4
используется ТОЛЬКО как оракул сверки (ADR-002), не как источник расчёта.

Парсинг читает:
* ``src/dftd4/reference.inc`` — поэлементные справочные данные
  (``clsq``, ``clsh``, ``refcovcn``, ``refcn``, ``refsys``, ``alphaiw``
  (23 точки), ``hcount``, ``ascale``, ``refn``) и справочные системы
  (``sscale``, ``secaiw`` (23 точки), ``secq``, ``seccn``);
* ``src/dftd4/data/{covrad,en,hardness,r4r2,zeff}.f90`` — атомные константы
  (118 элементов);
* ``assets/parameters.toml`` — параметры BJ-затухания (bj-eeq-atm) по
  функционалам (dftd4 v4.0.1; дубликаты — first-wins);
* ``multicharge/param/eeq2019.f90`` — таблицы модели зарядов EEQ
  (eeq_chi, eeq_eta, eeq_kcnchi, eeq_rad; 103 элемента) — стадия 3.

Источники закреплены в ``tools/dftd4_sources/`` (reference.inc + data/*.f90
из dftd4 main @2026-08-30; parameters.toml из dftd4 v4.0.1) и
``tools/multicharge_sources/`` (eeq2019.f90 из multicharge main @2026-08-31),
поэтому пересборка воспроизводима без сети:

    python tools/generate_d4_data.py            # из tools/dftd4_sources/
    python tools/generate_d4_data.py --src P --params P2   # другие источники

Скрипт пересобирает ``dispersion_d4_data.py``; значения записываются через
repr() (точный round-trip double). Полный текст provenance — в docstring
сгенерированного модуля.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re

OUT = pathlib.Path(__file__).resolve().parents[1] / "src/quantumlab/engine/dispersion_d4_data.py"

AATOA = 1.889726133890  # CODATA-2018 Angstrom -> au (сверка с оракулом — стадия 2)
MAX_ELEM = 118


def parse_fortran_array(
    data_dir: pathlib.Path, fname: str, varname: str
) -> tuple[list[float], bool]:
    """Read a 118-element Fortran array literal; return (values, uses_aatoau)."""
    text = (data_dir / fname).read_text()
    m = re.search(varname + r"\s*\(\s*max_elem\s*\)\s*=\s*(aatoau\s*\*?\s*)?\[", text)
    if not m:
        raise ValueError(f"{varname} not found in {fname}")
    uses_aatoau = bool(m.group(1))
    start = text.index("[", m.start())
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = text[start + 1 : end].replace("_wp", "")
    vals = [float(v) for v in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", body)]
    if len(vals) != MAX_ELEM:
        raise ValueError(f"{varname}: expected {MAX_ELEM} values, got {len(vals)}")
    return vals, uses_aatoau


def parse_eeq2019(path: pathlib.Path) -> dict[str, list[float]]:
    """Разбирает таблицы модели зарядов EEQ (multicharge param/eeq2019.f90).

    Четыре массива по ``max_elem = 103`` значений (``_wp``-суффиксы,
    фортрановские продолжения через ``&``): eeq_chi, eeq_eta, eeq_kcnchi,
    eeq_rad. Элементы Z > 103 в модели нет (get_eeq_* возвращают −1).
    """
    text = path.read_text()
    out: dict[str, list[float]] = {}
    for varname in ("eeq_chi", "eeq_eta", "eeq_kcnchi", "eeq_rad"):
        m = re.search(varname + r"\s*\(\s*max_elem\s*\)\s*=\s*\[", text)
        if not m:
            raise SystemExit(f"eeq2019: {varname} not found in {path}")
        start = text.index("[", m.start())
        depth, end = 0, start
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        body = text[start + 1 : end].replace("_wp", "")
        vals = [float(v) for v in re.findall(_NUM, body)]
        if len(vals) != 103:
            raise SystemExit(f"eeq2019: {varname} — 103 значения, получено {len(vals)}")
        out[varname] = vals
    return out


def logical_statements(text: str) -> list[str]:
    """Join `&` continuations, then split `;`-separated data statements.

    Returns one string per `data ... / ... /` statement.
    """
    text2 = re.sub(r"&\s*\n", " ", text)  # trailing & + newline
    text2 = re.sub(r"\n\s*&\s*", " ", text2)  # leading &
    statements = []
    for line in text2.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if part:
                statements.append(part)
    return statements


_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


def parse_reference_inc(inc_path: pathlib.Path) -> dict:
    """Parse reference.inc into per-element and per-reference-system tables."""
    text = inc_path.read_text()
    logical = logical_statements(text)

    refn = [0] * (MAX_ELEM + 1)
    clsq = {}  # (ir, num)
    clsh = {}
    refcovcn = {}
    refcn = {}
    refsys = {}
    hcount = {}
    ascale = {}
    alphaiw = {}  # (ir, num) -> list[23]
    sscale = {}  # isys
    secaiw = {}  # isys -> list[23]
    secq = {}
    seccn = {}
    per_ref = {
        "clsq": clsq,
        "clsh": clsh,
        "refcovcn": refcovcn,
        "refcn": refcn,
        "hcount": hcount,
        "ascale": ascale,
    }

    pat2 = re.compile(r"data\s+(\w+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/(.*?)/")
    pat1 = re.compile(r"data\s+(\w+)\s*\(\s*(\d+)\s*\)\s*/(.*?)/")
    pat23 = re.compile(r"data\s+(\w+)\s*\(\s*:\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*/(.*?)/")
    pat23b = re.compile(r"data\s+(\w+)\s*\(\s*:\s*,\s*(\d+)\s*\)\s*/(.*?)/")

    def nums(s: str) -> list[float]:
        return [float(v.replace("_wp", "")) for v in re.findall(_NUM, s.replace("_wp", ""))]

    for line in logical:
        if not line.lstrip().lower().startswith("data"):
            continue
        m = pat23.match(line)
        if m:  # alphaiw(:, ir, num)
            name, ir, num, body = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            vals = nums(body)
            if len(vals) == 23:
                alphaiw[(ir, num)] = vals
            continue
        m = pat23b.match(line)
        if m:  # secaiw(:, isys)
            name, isys, body = m.group(1), int(m.group(2)), m.group(3)
            vals = nums(body)
            if len(vals) == 23:
                secaiw[isys] = vals
            continue
        m = pat2.match(line)
        if m:  # name(ir, num)
            name, ir, num, body = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            vals = nums(body)
            if len(vals) != 1:
                continue
            v = vals[0]
            if name == "refsys":
                refsys[(ir, num)] = int(v)
            elif name in per_ref:
                per_ref[name][(ir, num)] = v
            continue
        m = pat1.match(line)
        if m:  # name(single)
            name, idx, body = m.group(1), int(m.group(2)), m.group(3)
            vals = nums(body)
            if len(vals) != 1:
                continue
            v = vals[0]
            if name == "refn":
                refn[idx] = int(v)
            elif name in ("sscale", "secq", "seccn"):
                if name == "sscale":
                    sscale[idx] = v
                elif name == "secq":
                    secq[idx] = v
                elif name == "seccn":
                    seccn[idx] = v
            continue

    return {
        "refn": refn,
        "clsq": clsq,
        "clsh": clsh,
        "refcovcn": refcovcn,
        "refcn": refcn,
        "refsys": refsys,
        "hcount": hcount,
        "ascale": ascale,
        "alphaiw": alphaiw,
        "sscale": sscale,
        "secaiw": secaiw,
        "secq": secq,
        "seccn": seccn,
    }


def parse_parameters_toml(path: pathlib.Path) -> dict:
    """Разбирает bj-eeq-atm параметры затухания по функционалам.

    Формат: секция ``[parameter.<xc>]`` содержит строку
    ``d4.bj-eeq-atm = { s8=..., a1=..., a2=..., [s6=...], [s9=...], [alp=...] }``.
    Дефолты из ``[default.parameter]``: s6=1.0, s9=1.0, alp=16.0.

    Смысл дубликата (напр. две строки bj-eeq-atm в [parameter.hse12]):
    **first-wins** — зафиксировано сверкой с libdft4 4.0.1 (hse12 == первой
    строке; повторные строки строка-секций с ``_`` не должны затенять
    предыдущую секцию, поэтому паттерн секции включает ``_``).
    """
    kv_pat = re.compile(r"(\w+)\s*=\s*([+-]?[0-9]+(?:\.[0-9]+)?)\b")
    line_pat = re.compile(r"d4\.bj-eeq-atm\s*=\s*\{([^}]*)\}")
    sec_pat = re.compile(r"\[parameter\.([a-z0-9_-]+)\]")
    out: dict = {}
    dups: list[str] = []
    cur = None
    for line in path.read_text().splitlines():
        line = line.strip()
        m = sec_pat.match(line)
        if m:
            cur = m.group(1)
            continue
        if cur is None:
            continue
        m = line_pat.search(line)
        if not m:
            continue
        kv = {k: float(v) for k, v in kv_pat.findall(m.group(1))}
        if not {"s8", "a1", "a2"} <= kv.keys():
            raise SystemExit(f"[parameter.{cur}]: bj-eeq-atm missing s8/a1/a2: {line}")
        if cur in out:  # дубликат внутри секции — first-wins (см. docstring)
            dups.append(cur)
            continue
        out[cur] = {
            "s6": kv.get("s6", 1.0),
            "s8": kv["s8"],
            "a1": kv["a1"],
            "a2": kv["a2"],
            "s9": kv.get("s9", 1.0),
            "alp": kv.get("alp", 16.0),
        }
    if dups:
        print(f"note: duplicate bj-eeq-atm keys (first-wins): {sorted(set(dups))}")
    return out


_SOURCES_DIR = pathlib.Path(__file__).resolve().parent / "dftd4_sources"
_EEQ_PATH = pathlib.Path(__file__).resolve().parent / "multicharge_sources/eeq2019.f90"


def _resolve_src(args_src: str | None) -> pathlib.Path:
    """Find the dftd4 ``src/dftd4`` layout (arg > $DFTD4_SRC > pinned sources).

    Закреплённые источники — ``tools/dftd4_sources/`` (reference.inc +
    data/*.f90 из dftd4 main @2026-08-30; сверка C6/ЧК с libdftd4 4.0.1 —
    tests/test_engine_d4.py).
    """
    candidates: list[str] = []
    if args_src:
        candidates.append(args_src)
    if env := os.environ.get("DFTD4_SRC"):
        candidates.append(env)
    candidates += [str(_SOURCES_DIR), "/tmp/dftd4-main/src/dftd4"]
    for c in candidates:
        p = pathlib.Path(c)
        if (p / "reference.inc").is_file():
            return p
    raise SystemExit(
        "dftd4 source not found. Pass --src /path/to/dftd4/src/dftd4 "
        "(or set $DFTD4_SRC) — see tools/generate_d4_data.py docstring."
    )


def main() -> None:
    """CLI entry: locate dftd4 source, parse, run checks, write the module."""
    global OUT
    ap = argparse.ArgumentParser(description="Regenerate D4 data module (see docstring).")
    ap.add_argument(
        "--src", help="path to dftd4 src/dftd4 directory (default: tools/dftd4_sources)"
    )
    ap.add_argument(
        "--params",
        default=str(_SOURCES_DIR / "parameters.toml"),
        help="dftd4 assets/parameters.toml (default: pinned v4.0.1 copy)",
    )
    ap.add_argument(
        "--eeq",
        default=str(_EEQ_PATH),
        help="multicharge param/eeq2019.f90 (default: pinned copy)",
    )
    ap.add_argument("--out", default=str(OUT), help="output data module path")
    args = ap.parse_args()

    src = _resolve_src(args.src)
    OUT = pathlib.Path(args.out)
    data_dir = src / "data"
    inc_path = src / "reference.inc"

    # atomic constants
    covalent, uses_a = parse_fortran_array(data_dir, "covrad.f90", "covalent_rad_2009")
    pauling_en, _ = parse_fortran_array(data_dir, "en.f90", "pauling_en")
    hardness, _ = parse_fortran_array(data_dir, "hardness.f90", "chemical_hardness")
    r4_over_r2, _ = parse_fortran_array(data_dir, "r4r2.f90", "r4_over_r2")
    zeff, _ = parse_fortran_array(data_dir, "zeff.f90", "effective_nuclear_charge")

    ref = parse_reference_inc(inc_path)

    param_path = pathlib.Path(args.params)
    if not param_path.is_file():
        raise SystemExit(f"parameters.toml not found: {param_path} (pass --params)")
    params = parse_parameters_toml(param_path)
    if not params:
        raise SystemExit(f"no damping parameters parsed from {param_path}")
    print("=== damping parameters ===")
    print(f"source: {param_path}  ({len(params)} functionals, bj-eeq-atm)")

    eeq_path = pathlib.Path(args.eeq)
    if not eeq_path.is_file():
        raise SystemExit(f"eeq2019.f90 not found: {eeq_path} (pass --eeq)")
    eeq = parse_eeq2019(eeq_path)
    print("=== EEQ charge model (multicharge eeq2019) ===")
    print(f"source: {eeq_path}  ({len(eeq['eeq_chi'])} elements)")
    print(
        f"  chi: H={eeq['eeq_chi'][0]:.8f} C={eeq['eeq_chi'][5]:.8f} "
        f"O={eeq['eeq_chi'][7]:.8f}"
    )
    print(f"  rad H={eeq['eeq_rad'][0]:.8f}  kcnchi O={eeq['eeq_kcnchi'][7]:.8f}")

    # ---- sanity checks ----
    print("=== atomic constants ===")
    print(f"covalent_rad C(6)={covalent[5]:.4f} A, H(1)={covalent[0]:.4f} A  (aatoau={uses_a})")
    print(f"zeff C(6)={zeff[5]:.1f}, hardness C(6)={hardness[5]:.4f}")

    print("\n=== reference data (per element) ===")
    # count elements with data
    n_with_ref = sum(1 for z in range(1, MAX_ELEM + 1) if ref["refn"][z] > 0)
    print(f"elements with refn>0: {n_with_ref}")
    for z, sym in [(1, "H"), (6, "C"), (7, "N"), (8, "O"), (9, "F")]:
        n = ref["refn"][z]
        refs = []
        for ir in range(1, n + 1):
            rs = ref["refsys"].get((ir, z), 0)
            refs.append(
                f"ir{ir}[cn={ref['refcovcn'].get((ir, z), 0):.3f} "
                f"clsq={ref['clsq'].get((ir, z), 0):+.3f} refsys={rs} "
                f"aiw0={ref['alphaiw'].get((ir, z), [0])[0]:.3f}]"
            )
        print(f"  {sym}(Z={z}) refn={n}  " + " ".join(refs))

    print("\n=== reference systems (isys) ===")
    for isys in sorted(ref["sscale"]):
        sa = ref["secaiw"].get(isys, [0])[0]
        sq = ref["secq"].get(isys, 0)
        print(f"  isys={isys}: sscale={ref['sscale'][isys]:.6f}  secaiw0={sa:.4f}  secq={sq:+.4f}")

    # ---- completeness check (gaps allowed: empty slots are refsys=0, skipped) ----
    # refn(Z) = highest reference index (array dim 7); a filled reference (ir,Z)
    # must carry all 7 fields and a refsys in the reference-atom set. Holes
    # (e.g. Z=92 ir=3) are legitimately absent and contribute 0.
    problems = []
    refsys_set = set(ref["secaiw"].keys())
    total_filled = 0
    total_holes = 0
    for z in range(1, MAX_ELEM + 1):
        n = ref["refn"][z]
        filled_ir = sorted(ir for (ir, zz) in ref["alphaiw"] if zz == z)
        if n == 0:
            if filled_ir:
                problems.append(f"Z={z} refn=0 but {len(filled_ir)} filled refs")
            continue
        if not filled_ir:
            problems.append(f"Z={z} refn={n} but no filled refs")
            continue
        for ir in filled_ir:
            key = (ir, z)
            for fld in (
                "clsq",
                "clsh",
                "refcovcn",
                "refcn",
                "refsys",
                "hcount",
                "ascale",
                "alphaiw",
            ):
                if key not in ref[fld]:
                    problems.append(f"Z={z} ir={ir} missing {fld}")
            aiw = ref["alphaiw"].get(key)
            if aiw is not None and len(aiw) != 23:
                problems.append(f"Z={z} ir={ir} alphaiw len {len(aiw)}")
            rs = ref["refsys"].get(key)
            if rs is not None and rs != 0 and rs not in refsys_set:
                problems.append(f"Z={z} ir={ir} refsys={rs} not in refsys set {sorted(refsys_set)}")
        if n < max(filled_ir):
            problems.append(f"Z={z} refn={n} < max filled ir={max(filled_ir)}")
        if n > 7:
            problems.append(f"Z={z} refn={n} > 7 (array dim)")
        total_filled += len(filled_ir)
        total_holes += n - len(filled_ir)
    n_problems = len(problems)
    print(f"\ncompleteness: {n_problems} problems | filled={total_filled} holes={total_holes}")
    for p in problems[:12]:
        print("  !", p)
    if problems:
        raise SystemExit("data incomplete — aborting module generation")

    _write_module(covalent, pauling_en, hardness, r4_over_r2, zeff, ref, params, eeq)


def _write_module(
    covalent: list[float],
    pauling_en: list[float],
    hardness: list[float],
    r4_over_r2: list[float],
    zeff: list[float],
    ref: dict,
    params: dict,
    eeq: dict[str, list[float]],
) -> None:
    """Emit the data module (compact, repr() values, `# fmt: off`)."""
    doc = '''"""Дисперсионная поправка DFT-D4: табличные данные модели.

ФАЙЛ ГЕНЕРИРУЕТСЯ — не править вручную. Источник: ``tools/generate_d4_data.py``
(извлекает из Fortran-таблиц референсной реализации dftd4, Grimme et al.,
J. Chem. Theory Comput. 2021, 17, 6579; mctc-lib / multicharge / dftd4).

Закреплённые источники:
* ``tools/dftd4_sources/`` — справочные данные (reference.inc, data/*.f90)
  dftd4 main, чек-аут 2026-08-30; параметры BJ-затухания (parameters.toml)
  dftd4 **v4.0.1** (версия libdftd4-оракула); дубликаты ключей — first-wins
  (семантика dftd4);
* ``tools/multicharge_sources/eeq2019.f90`` — таблицы модели зарядов EEQ
  (multicharge main @2026-08-31; Caldeweyher et al., J. Chem. Phys. 2019,
  150, 154122).

Библиотека libdftd4 используется ТОЛЬКО как оракул сверки (ADR-002), не как
источник расчёта: расчёт идёт по этим таблицам в чистом Python.
Сверка с libdftd4 4.0.1 — tests/test_engine_d4.py (C6, ЧК, энергия, EEQ).

Индексация по атомному номеру (1-based; индекс 0 не используется).
``D4_REF_SYS`` хранит атомный номер атома, определяющего справочную систему
(1=H, 2=He, 6=C, 7=N, 8=O, 9=F, 10=Ne, 11=Na, 17=Cl); пустые слоты (отсутствуют
в файле, напр. Z=92 ir=3) имеют refsys=0 и не вносят вклада.
Таблицы EEQ (D4_EEQ_*) содержат 103 элемента (Z=1..103); для Z>103 в
референсной модели зарядов значений нет.
Значения записаны через repr() — точный round-trip double.
"""'''

    def fnum(v: float) -> str:
        # repr() даёт точный round-trip double — совпадение с Fortran-оракулом.
        return repr(v)

    def wrap_width(items: list[str], indent: str, maxw: int = 88) -> list[str]:
        """Разбивает строки чисел на строки шириной <= maxw (с запятыми)."""
        lines: list[str] = []
        cur = ""
        for s in items:
            if not cur:
                cur = s
            elif len(cur) + 2 + len(s) <= maxw:
                cur = cur + ", " + s
            else:
                lines.append(cur)
                cur = s
        lines.append(cur)
        return [indent + ln + "," for ln in lines]

    out_lines: list[str] = []

    def array_block(name: str, vals: list[float], comment: str = "") -> None:
        head = f"{name} = ["
        if comment:
            head += f"  # {comment}"
        out_lines.append(head)
        out_lines.extend(wrap_width([fnum(v) for v in vals], "    "))
        out_lines.append("]")
        out_lines.append("")

    def scalar_dict_block(name: str, d: dict, as_int: bool = False) -> None:
        out_lines.append(f"{name} = {{")
        for k, v in sorted(d.items()):
            val = repr(v) if as_int else fnum(v)
            out_lines.append(f"    {k!r}: {val},")
        out_lines.append("}")
        out_lines.append("")

    def list_dict_block(name: str, d: dict) -> None:
        out_lines.append(f"{name} = {{")
        for k, v in sorted(d.items()):
            out_lines.append(f"    {k!r}: [")
            out_lines.extend(wrap_width([fnum(x) for x in v], "        "))
            out_lines.append("    ],")
        out_lines.append("}")
        out_lines.append("")

    out_lines.append(doc)
    out_lines.append("")
    out_lines.append("from __future__ import annotations")
    out_lines.append("")
    out_lines.append("# region: табличные данные (генерируемые, формат зафиксирован)")
    out_lines.append("# fmt: off")
    out_lines.append("")
    out_lines.append(f"AATOA = {AATOA!r}  # Å -> а.е. (CODATA-2018); сверка с оракулом — стадия 2")
    out_lines.append("")

    out_lines.append("# --- Атомные константы (118 элементов, 1-based) ---")
    array_block("COVALENT_RAD_ANGSTROM", covalent, "ковалентный радиус, Å")
    array_block("PAULING_EN", pauling_en, "электронегативность Полинга")
    array_block("CHEMICAL_HARDNESS", hardness, "химическая твёрдость")
    array_block("R4_OVER_R2", r4_over_r2, "средний радиус 4-го порядка / 2-го")
    array_block("ZEFF", zeff, "эффективный заряд ядра")

    out_lines.append("# --- Модель зарядов EEQ (multicharge eeq2019; 103 элемента, 1-based) ---")
    array_block("D4_EEQ_CHI", eeq["eeq_chi"], "электронегативность EEQ")
    array_block("D4_EEQ_ETA", eeq["eeq_eta"], "химическая твёрдость EEQ")
    array_block("D4_EEQ_KCNCHI", eeq["eeq_kcnchi"], "CN-масштабирование χ")
    array_block("D4_EEQ_RAD", eeq["eeq_rad"], "ширина гауссова заряда (bohr)")

    out_lines.append("# --- Справочные системы D4 (ключ = атомный номер атома-референса) ---")
    scalar_dict_block("REF_SYSSCALE", ref["sscale"])
    list_dict_block("REF_SYSAIW", ref["secaiw"])

    out_lines.append("# --- Поэлементные справочные данные (ключ (ir, Z)) ---")
    out_lines.append("D4_NREF = [  # число справочных систем (max ir), 1-based")
    out_lines.extend(wrap_width([str(v) for v in ref["refn"][1 : MAX_ELEM + 1]], "    "))
    out_lines.append("]")
    out_lines.append("")
    scalar_dict_block("D4_REF_COV_CN", ref["refcovcn"])
    scalar_dict_block("D4_CLS_Q", ref["clsq"])
    scalar_dict_block("D4_CLS_H", ref["clsh"])
    scalar_dict_block("D4_REF_CN", ref["refcn"])
    scalar_dict_block("D4_REF_SYS", {k: int(v) for k, v in ref["refsys"].items()}, as_int=True)
    scalar_dict_block("D4_HCOUNT", ref["hcount"])
    scalar_dict_block("D4_ASCALE", ref["ascale"])
    list_dict_block("D4_ALPHA_IW", ref["alphaiw"])

    out_lines.append("# --- Параметры BJ-затухания (bj-eeq-atm): (s6, s8, a1, a2, s9, alp) ---")
    out_lines.append("D4_DAMPING_PARAMS = {")
    for xc in sorted(params):
        p = params[xc]
        out_lines.append(
            f"    {xc!r}: ({fnum(p['s6'])}, {fnum(p['s8'])}, {fnum(p['a1'])}, "
            f"{fnum(p['a2'])}, {fnum(p['s9'])}, {fnum(p['alp'])}),"
        )
    out_lines.append("}")
    out_lines.append("")
    out_lines.append("# fmt: on")
    out_lines.append("# endregion")

    out = "\n".join(out_lines) + "\n"
    OUT.write_text(out, encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(out)} bytes, {len(out_lines)} lines)")


if __name__ == "__main__":
    main()
