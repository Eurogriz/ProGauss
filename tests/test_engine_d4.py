"""Сверка ядра DFT-D4 (стадии 2–3) с независимой сборкой libdftd4.

По ADR-002 libdftd4 используется **только как оракул**: движок не читает
его таблицы и не вызывает его рутин в production. Таблицы модели (стадия 1)
извлечены из исходников dftd4 (``tools/dftd4_sources/``, ``tools/multicharge_sources/``,
см. docstring ``dispersion_d4_data``); совпадение с независимой
компиляцией модели — реальная верификация, а не самопроверка.

Оракул: ``pyscf.dispersion.dftd4`` (libdftd4 4.0.1, версия фиксируется
тестом ``test_oracle_version_is_4_0_1``). Версия данных: справочные —
dftd4 main @2026-08-30, параметры — dftd4 v4.0.1, EEQ — multicharge
main @2026-08-31; всё проверено оракулом на C6/ЧК/энергии/зарядах.

Область: стадии 2–4 — гомоядерные диатомы (энергия; заряды на них точны
по симметрии), гетероядерные молекулы/ионы (EEQ-заряды, ЧК, C6), полная
энергия молекул с nat>2 (включая трёхчленный C9-вклад) и аналитический
градиент D4 (явная геометрия + связь по C6/ЧК + CP-решение dq/dR).

Погрешность: PySCF передаёт в libdftd4 координаты, пересчитанные из Å
своей константой (1.889726124564), а наш движок получает bohr напрямую
(наша константа = ``aatoau`` dftd4 = 1.889726133890; относительная
разница ~5e-9). На малых r расхождение ~1e-13 э.е., на больших растёт
как 6·Δr/r (до ~2e-10); через ЧК на заряды накладывается та же разница
(до ~1e-8 e). Поэтому сверка — **относительным** допуском 1e-6 (для
энергии и C6) и абсолютным 1e-7 (для зарядов — у симметричных атомов
заряд нулевой, относительный допуск не применим), как в
``test_crosscheck_dftd3``.

Регрессионные тесты данных фиксируют ошибки парсеров: паттерн секций
без ``_`` стадии 1 (``test_damping_params_*``) и точность таблиц EEQ
(``test_eeq_tables_regression``).
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from quantumlab.engine import dispersion_d4 as d4
from quantumlab.engine import dispersion_d4_data as d4d
from quantumlab.engine import eeq_charges as eeq

pyscf = pytest.importorskip("pyscf", reason="pyscf-dispersion нужен только для независимой сверки")
#: Подмодуль регистрируется через importlib: статический импорт заставил бы
#: mypy искать у PySCF заглушки типов, которых нет.
_D4_MODULE = importlib.import_module("pyscf.dispersion.dftd4")

pytestmark = pytest.mark.scientific

#: Константа Å→bohr — та же, что ``aatoau`` в dftd4 (CODATA-2018).
AATOA = 1.889726133890

#: Относительный допуск сверки с оракулом (разница констант Å→bohr ~5e-9).
REL_TOL = 1e-6

#: (символ, Z, равновесное расстояние, Å). H2/N2/O2/F2 — план стадии 2;
#: C2 (7 справочных систем, refsys He) и Li2 (3) — расширение покрытия.
DIATOMICS: list[tuple[str, int, float]] = [
    ("H", 1, 0.7414),
    ("N", 7, 1.0977),
    ("O", 8, 1.2075),
    ("F", 9, 1.4119),
    ("C", 6, 1.2425),
    ("Li", 3, 2.673),
]

#: Функционалы покрытия: GGA с s6=1 (pbe, blyp, tpss), гибриды (pbe0, hse12
#: — дубликат ключа first-wins), RSH с отрицательным s8 (wb97x-2008).
XCS: list[str] = ["pbe", "blyp", "tpss", "pbe0", "hse12", "wb97x-2008"]

STRETCHES: tuple[float, ...] = (1.0, 1.3, 1.6)


def _pos_bohr(r_angstrom: float) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_angstrom * AATOA]])


def _oracle_energy(sym: str, r_angstrom: float, xc: str) -> float:
    """Энергия D4 (bj-eeq-atm, EEQ q=0 для гомоядра) из libdftd4, э.е."""
    mol = pyscf.gto.Mole(
        atom=f"{sym} 0 0 0; {sym} 0 0 {r_angstrom:.6f}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mol.build()
    disp = pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc=xc)
    return float(disp.get_dispersion()["energy"])


#: (Z, атомные номера, координаты Å) — экспериментальные геометрии.
MOLECULES: dict[str, tuple[list[int], list[list[float]]]] = {
    # O-H 0.9572, угол 104.5°
    "H2O": ([8, 1, 1], [[0.0, 0.0, 0.0], [0.7572, 0.5858, 0.0], [-0.7572, 0.5858, 0.0]]),
    # N-H 1.012
    "NH3": (
        [7, 1, 1, 1],
        [
            [0.0, 0.0, 0.023],
            [0.0, 0.937, -0.277],
            [0.812, -0.544, -0.277],
            [-0.812, -0.544, -0.277],
        ],
    ),
    # C=O 1.16
    "CO2": ([8, 6, 8], [[0.0, 0.0, 1.16], [0.0, 0.0, 0.0], [0.0, 0.0, -1.16]]),
    # C-F 1.392, C-H 1.087
    "CH3F": (
        [6, 9, 1, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.392],
            [1.026, -0.465, -0.464],
            [-0.513, 0.892, -0.464],
            [-0.513, -0.427, -0.464],
        ],
    ),
    # C-C 1.341, C-H 1.088
    "C2H4": (
        [6, 6, 1, 1, 1, 1],
        [
            [-0.6705, 0.0, 0.0],
            [0.6705, 0.0, 0.0],
            [-1.240, 0.928, 0.0],
            [-1.240, -0.928, 0.0],
            [1.240, 0.928, 0.0],
            [1.240, -0.928, 0.0],
        ],
    ),
    # C=O 1.206, C-H 1.041
    "H2CO": (
        [6, 8, 1, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.206], [0.948, 0.0, -0.467], [-0.948, 0.0, -0.467]],
    ),
    "HCl": ([1, 17], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2746]]),
    # ионная пара
    "NaCl": ([11, 17], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.361]]),
}

#: (название, Z, координаты, полный заряд) — ионы.
IONS: dict[str, tuple[list[int], list[list[float]], int]] = {
    "NaCl+": ([11, 17], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.361]], 1),
    "Cl-": ([17], [[0.0, 0.0, 0.0]], -1),
    "H2O+": ([8, 1, 1], [[0.0, 0.0, 0.0], [0.7572, 0.5858, 0.0], [-0.7572, 0.5858, 0.0]], 1),
}

#: Допуски сверки стадии 3 (см. docstring модуля).
CHARGE_TOL = 1e-7
CN_TOL = 1e-8
C6_TOL = 1e-6


def _oracle_props(
    zs: list[int], pos_angstrom: list[list[float]], xc: str, charge: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ЧК, заряды, C6) из libdftd4 через dftd4_get_properties."""
    syms = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 17: "Cl"}
    atom = [(syms[z], [float(x) for x in p]) for z, p in zip(zs, pos_angstrom, strict=True)]
    n_electrons = sum(zs) - charge
    mol = pyscf.gto.Mole(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        charge=charge,
        spin=1 if n_electrons % 2 else 0,
        verbose=0,
    )
    mol.build()
    disp = pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc=xc)
    lib = _D4_MODULE.libdftd4
    nat = mol.natm
    err = lib.dftd4_new_error()
    cn = np.zeros(nat)
    ch = np.zeros(nat)
    c6 = np.zeros((nat, nat))
    alpha = np.zeros(nat)
    lib.dftd4_get_properties(
        err, disp._mol, disp._disp, cn.ctypes, ch.ctypes, c6.ctypes, alpha.ctypes
    )
    return cn, ch, c6


def test_oracle_version_is_4_0_1() -> None:
    # Данные зафиксированы под libdftd4 4.0.1 (см. docstring модуля); если
    # pyscf перепакует другую версию, сверка потеряет смысл — тест явно это
    # покажет.
    lib = _D4_MODULE.libdftd4
    version = int(lib.dftd4_get_version())
    assert (version // 10000, (version // 100) % 100) == (4, 0)


@pytest.mark.parametrize("sym, z, r_eq", DIATOMICS)
@pytest.mark.parametrize("xc", XCS)
@pytest.mark.parametrize("stretch", STRETCHES)
def test_d4_energy_matches_oracle(sym: str, z: int, r_eq: float, xc: str, stretch: float) -> None:
    r = r_eq * stretch
    e_ref = _oracle_energy(sym, r, xc)
    model = d4.DFTD4([z, z], _pos_bohr(r), xc)
    e_ours = model.energy()
    assert abs(e_ours - e_ref) <= REL_TOL * abs(e_ref), (
        f"{sym}2 {xc} R={r:.3f} Å: ours={e_ours:.12e} oracle={e_ref:.12e} "
        f"d={abs(e_ours - e_ref):.3e}"
    )


def test_h2_cn_matches_oracle() -> None:
    # CN(H2, 0.7414 Å) = 0.90036174 (libdftd4 4.0.1). Без (4/3)-масштаба
    # радиусов и EN-фактора получается 0.046 — тест ловит обе ошибки.
    model = d4.DFTD4([1, 1], _pos_bohr(0.7414), "pbe")
    cn = model.coordination_numbers()
    assert abs(cn[0] - 0.90036174) < 1e-6
    # На вытянутой связи CN экспоненциально падает к 0.
    model_far = d4.DFTD4([1, 1], _pos_bohr(1.186), "pbe")
    assert abs(model_far.coordination_numbers()[0] - 0.00001741) < 1e-7


def test_c6_matrix_symmetric_and_saturates() -> None:
    model = d4.DFTD4([7, 7], _pos_bohr(1.0977), "pbe")
    c6 = model.c6_matrix()
    assert c6.shape == (2, 2)
    assert abs(c6[0, 1] - c6[1, 0]) == 0.0
    assert c6[0, 0] > 0.0
    # На большом расстоянии CN→0: C6 насыщается значением CN=0-системы и
    # перестаёт зависеть от r. На 3 bohr CN ещё ~0.9 (вклад средневзвешенной
    # системы), поэтому сходим именно в асимптотике (10 и 20 bohr, CN=0).
    c6_far = d4.DFTD4([7, 7], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]), "pbe").c6_matrix()
    c6_farther = d4.DFTD4([7, 7], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]), "pbe").c6_matrix()
    assert abs(c6_far[0, 1] - c6_farther[0, 1]) < 1e-8


def test_get_dispersion_result_consistent() -> None:
    model = d4.DFTD4([1, 1], _pos_bohr(0.7414), "pbe")
    res = model.get_dispersion()
    assert abs(res.energy_hartree - model.energy()) < 1e-15
    assert len(res.cn) == 2
    assert abs(res.c6[0][1] - res.c6[1][0]) < 1e-15
    # Гомоядерный димер: EEQ-заряды нулевые по симметрии (стадия 3).
    assert all(abs(q) < 1e-12 for q in res.charges)
    assert abs(sum(res.charges)) < 1e-12


# --- Стадия 3: EEQ-заряды (eeq2019) ---


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_eeq_charges_match_oracle(name: str) -> None:
    zs, pos = MOLECULES[name]
    _cn_ref, ch_ref, _c6_ref = _oracle_props(zs, pos, "pbe")
    model = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe")
    q = np.array(model.q)
    assert max(abs(a - b) for a, b in zip(q, ch_ref, strict=True)) < CHARGE_TOL, (
        f"{name}: ours={np.round(q, 6)} oracle={np.round(ch_ref, 6)}"
    )


@pytest.mark.parametrize("name", sorted(IONS))
def test_eeq_ion_charges_match_oracle(name: str) -> None:
    zs, pos, charge = IONS[name]
    _cn_ref, ch_ref, _c6_ref = _oracle_props(zs, pos, "pbe", charge)
    model = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe", total_charge=charge)
    q = np.array(model.q)
    assert max(abs(a - b) for a, b in zip(q, ch_ref, strict=True)) < CHARGE_TOL, (
        f"{name}: ours={np.round(q, 6)} oracle={np.round(ch_ref, 6)}"
    )
    # Заряды обязаны давать полный заряд (ограничение Σq = Q).
    assert abs(q.sum() - charge) < 1e-12


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_eeq_cn_and_c6_match_oracle(name: str) -> None:
    # ЧК модели C6 (erf_dftd4, EN-фактор) и C6-матрица — через оракул.
    zs, pos = MOLECULES[name]
    cn_ref, _ch_ref, c6_ref = _oracle_props(zs, pos, "pbe")
    model = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe")
    cn = model.coordination_numbers()
    c6 = model.c6_matrix(cn)
    assert max(abs(a - b) for a, b in zip(cn, cn_ref, strict=True)) < CN_TOL, (
        f"{name} CN: ours={np.round(cn, 6)} oracle={np.round(cn_ref, 6)}"
    )
    assert np.max(np.abs(c6 - c6_ref)) < C6_TOL, (
        f"{name} C6: ours=\n{np.round(c6, 4)}\noracle=\n{np.round(c6_ref, 4)}"
    )


def test_eeq_charges_sum_to_total() -> None:
    # Σq = Q для нейтральных молекул и ионов.
    for zs, pos in MOLECULES.values():
        q = np.array(d4.DFTD4(zs, np.array(pos) * AATOA, "pbe").q)
        assert abs(q.sum()) < 1e-12
    for zs, pos, charge in IONS.values():
        q = np.array(d4.DFTD4(zs, np.array(pos) * AATOA, "pbe", total_charge=charge).q)
        assert abs(q.sum() - charge) < 1e-12


def test_eeq_explicit_charges_override() -> None:
    # Явные заряды перекрывают EEQ (стадия 2: q=0 на диатомах).
    zs, pos = MOLECULES["H2O"]
    m_eeq = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe")
    m_zero = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe", charges=[0.0, 0.0, 0.0])
    # C6 с q=0 отличаются от C6 с EEQ-зарядами (zeta-масштабирование).
    c6_eeq = m_eeq.c6_matrix()
    c6_zero = m_zero.c6_matrix()
    assert np.max(np.abs(c6_eeq - c6_zero)) > 1e-6


def test_eeq_homonuclear_zero_charge() -> None:
    # На гомоядерных диатомах EEQ даёт точный 0 (совпадение со стадией 2).
    for _sym, z, r in DIATOMICS:
        q = d4.DFTD4([z, z], _pos_bohr(r), "pbe").q
        assert all(abs(x) < 1e-12 for x in q)


def test_eeq_z_beyond_table_rejected() -> None:
    # Таблицы eeq2019 до Z=103 (Np); дальше — ValueError, а не молча −1.
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * AATOA]])
    eeq.EEQ2019([103, 1], pos)  # 103 — граница таблицы, допустима
    with pytest.raises(ValueError, match=r"вне области 1\.\.103"):
        eeq.EEQ2019([104, 1], pos)
    # DFTD4 отклоняет 104 раньше — Rf–Rg не поддерживаются моделью D4.
    with pytest.raises(ValueError, match="не поддерживается"):
        d4.DFTD4([104, 1], pos, "pbe")


def test_eeq_bad_positions_rejected() -> None:
    with pytest.raises(ValueError, match="размерность"):
        eeq.EEQ2019([1, 1], np.array([[0.0, 0.0, 0.0]]))


def test_eeq_tables_regression() -> None:
    # Регрессия генератора: точные значения таблиц eeq2019 (103 элемента).
    assert len(d4d.D4_EEQ_CHI) == 103
    assert len(d4d.D4_EEQ_RAD) == 103
    # H(1), C(6), O(8): spot-значения из eeq2019.f90.
    assert d4d.D4_EEQ_CHI[0] == 1.23695041
    assert d4d.D4_EEQ_CHI[5] == 1.40028282
    assert d4d.D4_EEQ_CHI[7] == 1.56866440
    assert d4d.D4_EEQ_RAD[0] == 0.55159092
    assert d4d.D4_EEQ_RAD[5] == 1.88862966
    assert d4d.D4_EEQ_KCNCHI[7] == 0.11689703
    assert d4d.D4_EEQ_ETA[0] == -0.35015861
    # Np(103) — последний; Pu(104) отсутствует.
    assert d4d.D4_EEQ_CHI[102] == 1.13714110


def test_eeq_charge_model_constants() -> None:
    # Константы модели (multicharge new_eeq2019_model) — не путать с C6.
    assert eeq.EEQ_KCN == 7.5
    assert eeq.EEQ_CN_CUTOFF == 25.0  # 25 bohr — НЕ 30 (отсечение C6)
    assert eeq.EEQ_CN_MAX == 8.0


def test_eeq_cn_log_cut() -> None:
    # Плавное ограничение ЧК (mctc log_cn_cut): не min, а лог-сглаживание.
    # При CN=0 ограничение тождественно 0.
    assert eeq.cn_log_cut(0.0) == 0.0
    # Асимптота — ln(1+e^cnmax) = cnmax + ln(1+e^−cnmax) ≈ 8.0003354
    # ("Maximum CN, not strictly obeyed" — комментарий в mctc).
    asymptote = math.log1p(math.exp(8.0))
    assert eeq.cn_log_cut(100.0) == pytest.approx(asymptote, abs=1e-9)
    assert asymptote - 8.0 < 4e-4
    # При CN=8 значение заметно меньше (плавный «перегиб»).
    assert eeq.cn_log_cut(8.0) == pytest.approx(8.00033540637 - 0.69314718056, abs=1e-6)
    # Монотонность и мягкость при CN≪8.
    assert eeq.cn_log_cut(5.0) < eeq.cn_log_cut(10.0)
    assert eeq.cn_log_cut(1.0) == pytest.approx(1.0, abs=0.01)
    assert eeq.cn_log_cut(1.0) < 1.0


def test_damping_params_underscore_sections_regression() -> None:
    # Паттерн секций стадии 1 не знал ``_``: значения [parameter.pbe0_dh],
    # [parameter.pbe0_2], [parameter.dftb_*] затеняли предыдущие секции.
    assert d4d.D4_DAMPING_PARAMS["pbe0"] == (1.0, 1.20065498, 0.40085597, 5.02928789, 1.0, 16.0)
    assert d4d.D4_DAMPING_PARAMS["pbe0_dh"] == (
        0.875,
        0.96811578,
        0.47592488,
        5.08622873,
        1.0,
        16.0,
    )
    assert d4d.D4_DAMPING_PARAMS["pbe0_2"] == (0.5, 0.64299082, 0.76542115, 5.78578675, 1.0, 16.0)
    assert d4d.D4_DAMPING_PARAMS["opbe"] == (1.0, 3.06917417, 0.68267534, 2.22849018, 1.0, 16.0)
    assert d4d.D4_DAMPING_PARAMS["revdodpbep86"] == (0.5552, 0.0, 0.44, 3.6, 1.0, 16.0)
    assert "dftb_3ob" in d4d.D4_DAMPING_PARAMS
    assert len(d4d.D4_DAMPING_PARAMS) == 118


def test_hse12_duplicate_key_first_wins() -> None:
    # В parameters.toml v4.0.1 у hse12 две строки bj-eeq-atm; dftd4 берёт
    # ПЕРВУЮ (проверено: оракул воспроизводит параметры первой строки).
    assert d4d.D4_DAMPING_PARAMS["hse12"] == (1.0, 1.23500792, 0.39226921, 5.22036266, 1.0, 16.0)


def test_r2scan_3c_s9_parsed() -> None:
    # Единственный функционал с явным s9≠1 в v4.0.1 (s9=2.0).
    assert d4d.D4_DAMPING_PARAMS["r2scan-3c"][4] == 2.0


def test_unknown_functional_rejected() -> None:
    with pytest.raises(ValueError, match="нет обученных параметров"):
        d4.DFTD4([1, 1], _pos_bohr(0.7414), "not-a-functional")


def test_unsupported_element_rejected() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * AATOA]])
    with pytest.raises(ValueError, match="не поддерживается"):
        d4.DFTD4([104, 1], pos, "pbe")


def test_z_out_of_range_rejected() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * AATOA]])
    with pytest.raises(ValueError, match="вне области"):
        d4.DFTD4([0, 1], pos, "pbe")


def test_bad_positions_rejected() -> None:
    with pytest.raises(ValueError, match="размерность"):
        d4.DFTD4([1, 1], np.array([[0.0, 0.0, 0.0]]), "pbe")


# --- Стадия 4: энергия молекул (с C9) и аналитический градиент ---

#: Допуск градиента: libdft4 возвращает э.е./bohr; на наших системах
#: |g|~1e-4..1e-5 э.е./bohr, расхождение констант Å→bohr даёт ~1e-12.
GRAD_TOL = 1e-8


def _oracle_gradient(zs: list[int], pos_angstrom: list[list[float]], xc: str) -> np.ndarray:
    """Градиент D4 из libdft4 (э.е./bohr) через get_dispersion(grad=True)."""
    syms = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 17: "Cl"}
    atom = [(syms[z], [float(x) for x in p]) for z, p in zip(zs, pos_angstrom, strict=True)]
    mol = pyscf.gto.Mole(atom=atom, basis="sto-3g", unit="Angstrom", verbose=0)
    mol.build()
    disp = pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc=xc)
    return np.array(disp.get_dispersion(grad=True)["gradient"])


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_d4_molecule_energy_with_c9_matches_oracle(name: str) -> None:
    # nat>2: полная энергия (пары + C9) при фактических EEQ-зарядах.
    zs, pos = MOLECULES[name]
    e_ref = _oracle_energy_mol(zs, pos, "pbe")
    e_ours = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe").energy()
    assert abs(e_ours - e_ref) <= REL_TOL * abs(e_ref), (
        f"{name}: ours={e_ours:.12e} oracle={e_ref:.12e}"
    )


def _oracle_energy_mol(zs: list[int], pos_angstrom: list[list[float]], xc: str) -> float:
    """Энергия D4 для многоатомной системы из libdft4, э.е."""
    syms = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 17: "Cl"}
    atom = [(syms[z], [float(x) for x in p]) for z, p in zip(zs, pos_angstrom, strict=True)]
    mol = pyscf.gto.Mole(atom=atom, basis="sto-3g", unit="Angstrom", verbose=0)
    mol.build()
    disp = pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc=xc)
    return float(disp.get_dispersion()["energy"])


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_d4_gradient_matches_oracle(name: str) -> None:
    zs, pos = MOLECULES[name]
    g_ref = _oracle_gradient(zs, pos, "pbe")
    g_ours = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe").gradient()
    assert g_ours.shape == (len(zs), 3)
    err = np.max(np.abs(g_ours - g_ref))
    assert err < GRAD_TOL, f"{name}: max|Δg|={err:.3e}\nours={g_ours}\nref={g_ref}"


@pytest.mark.parametrize("name", sorted(IONS))
def test_d4_ion_gradient_matches_oracle(name: str) -> None:
    zs, pos, charge = IONS[name]
    syms = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 17: "Cl"}
    atom = [(syms[z], [float(x) for x in p]) for z, p in zip(zs, pos, strict=True)]
    n_electrons = sum(zs) - charge
    mol = pyscf.gto.Mole(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        charge=charge,
        spin=1 if n_electrons % 2 else 0,
        verbose=0,
    )
    mol.build()
    disp = pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc="pbe")
    g_ref = np.array(disp.get_dispersion(grad=True)["gradient"])
    g_ours = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe", total_charge=charge).gradient()
    err = np.max(np.abs(g_ours - g_ref))
    assert err < GRAD_TOL, f"{name}: max|Δg|={err:.3e}\nours={g_ours}\nref={g_ref}"


def test_d4_gradient_translational_invariance() -> None:
    # Градиент обязан давать нулевую силу при трансляции всей системы:
    # Σ_a g_a = 0 (в пределах машинной точности) — независимый от оракула
    # контроль, ловящий знак/индексные ошибки в сборке.
    for _name, (zs, pos) in MOLECULES.items():
        g = d4.DFTD4(zs, np.array(pos) * AATOA, "pbe").gradient()
        assert np.max(np.abs(g.sum(axis=0))) < 1e-10


def test_d4_gradient_finite_difference_matches() -> None:
    # Независимый контроль без оракула: аналитический градиент vs центральные
    # конечные разности энергии (шаг 1e-4 bohr, 2-й порядок точности).
    zs, pos = MOLECULES["H2O"]
    p0 = np.array(pos) * AATOA
    g_ours = d4.DFTD4(zs, p0, "pbe").gradient()
    h = 1e-4
    for a in range(len(zs)):
        for ic in range(3):
            pp = p0.copy()
            pm = p0.copy()
            pp[a, ic] += h
            pm[a, ic] -= h
            fd = (d4.DFTD4(zs, pp, "pbe").energy() - d4.DFTD4(zs, pm, "pbe").energy()) / (2 * h)
            assert abs(fd - g_ours[a, ic]) < 1e-7, (
                f"H2O atom {a} dir {ic}: fd={fd:.6e} analytic={g_ours[a, ic]:.6e}"
            )
