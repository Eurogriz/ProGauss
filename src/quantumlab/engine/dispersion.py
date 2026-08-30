"""Дисперсионная поправка DFT-D3: BJ- и zero-затухание, аналитический градиент.

Модель: двухчленная дисперсионная поправка

    E_D3 = −Σ_{i<j} C6^{(CN)}_{ij} · edisp(r_ij),

где ``C6^{(CN)}`` интерполируется по координационному числу (CN) атомов
взвешенным усреднением по табличным значениям для систем отсчёта, а
``edisp`` — затухающий множитель (BJ-рациональная функция или zero-затухание).
Действует на пару атомов, поэтому поправка масштабируется как O(N²) и не
зависит от базиса: она добавляется к энергии и градиенту любого SCF-метода
(HF, RKS) и не меняет электронную структуру — пересчёта плотности не нужно.

Происхождение формул и данных: s-dftd3 v1.2.1 (dftd3/simple-dftd3,
LGPL-3.0) — ``dftd3/ncoord.f90`` (CN и его производные), ``dftd3/model.f90``
(веса CN-интерполяции, C6-пары), ``dftd3/damping/bj.f90`` и
``dftd3/damping/zero.f90`` (затухание), ``dftd3/param.f90`` (параметры),
``dftd3/reference.f90`` + ``dftd3/data/*.f90`` (таблицы). Таблицы
перенесены в :mod:`quantumlab.engine.dispersion_data` генератором
``tools/generate_d3_data.py`` (флаг ``--verify-src`` перепарсит исходники
и сверяет каждое значение).

Сверка (требование §54 ТЗ — нет имитации результата): реализация сверяется
тестом с независимой сборкой s-dftd3 (pyscf-dispersion 1.5.0,
s-dftd3 @ 86dcf336 + mctc-lib v0.4.1) в ``tests/test_crosscheck_dftd3.py``:
расхождение энергии и сил ≤ 1e-6 (абсолютно) ограничено разницей константы
преобразования Å→bohr (AATOA_D3 = 1.8897261246 против нашей
1.8897259885789, относит. 7e-8); внутренний тест проверяет градиент
конечными разностями до 1e-10.

Область применения (честно, §54 ТЗ): элементы H, B, C, N, O, F, Si, P, S,
Cl, Br, I; функционалы с обученными параметрами — hf, pbe, pbe0, blyp,
b3lyp. Для LDA (svwn) параметров не существует, поэтому DFT-D3 с ним
отмечается как недоступный, а не приближается. Другие элементы и
функционалы отклоняются ValueError до начала расчёта.

Сложность: O(N²) по парам атомов с отсечением 60 bohr (энергия) и 40 bohr
(CN). Чистый Python: на молекуле в 100 атомов поправка — доли секунды,
существенно дешевле самого SCF. Векторизация по парам рассматривается при
профилировании (§52 ТЗ), а не наугад.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import DispersionCorrection
from quantumlab.engine import dispersion_data as dd
from quantumlab.engine.constants import angstrom_to_bohr

__all__ = [
    "DispersionContribution",
    "dftd3_contribution",
    "dftd3_functionals",
]

#: Квадрат пространственного отсечения (bohr²) для парной энергии.
#: s-dftd3: default_rc = 60 bohr. Пары дальше не дают вклада: затухающий
#: множитель убывает как 1/r⁶, на 60 bohr он ~1e-36.
_DISPERSION_CUTOFF_SQ: float = 60.0**2

#: Квадрат отсечения (bohr²) для подсчёта координационного числа.
#: s-dftd3: default_rc_ncoord = 40 bohr.
_COORDINATION_CUTOFF_SQ: float = 40.0**2

#: Пары атомов ближе этого квадрата (bohr²) пропускаются: это наложение
#: ядер, где модель не определена, а деление на r даёт численный шум.
_MIN_DISTANCE_SQ: float = 1e-12


@dataclass(frozen=True)
class DispersionContribution:
    """Вклад поправки D3 в энергию и ядерный градиент.

    ``energy_hartree`` — полная D3-энергия (отрицательная), хартри;
    ``gradient`` — вклад D3 в градиент энергии, форма ``(n_atoms, 3)``,
    хартри/bohr, ориентирован так же, как градиент SCF (движок складывает
    их без преобразований).
    """

    model: str
    functional: str
    energy_hartree: float
    gradient: np.ndarray


def dftd3_functionals(correction: DispersionCorrection) -> tuple[str, ...]:
    """Функционалы с обученными параметрами D3 (по порядку таблицы)."""
    if correction is DispersionCorrection.D3_BJ:
        return tuple(dd.BJ_PARAMS)
    if correction is DispersionCorrection.D3_ZERO:
        return tuple(dd.ZERO_PARAMS)
    msg = f"Неизвестная дисперсионная поправка: {correction!r}"
    raise ValueError(msg)


def _functional_name(correction: DispersionCorrection, functional: str | None) -> str:
    """Имя функционала для таблицы параметров; ``None`` — это HF."""
    name = "hf" if functional is None else functional
    known = dftd3_functionals(correction)
    if name not in known:
        msg = (
            f"D3 ({correction.value}) не обучен для функционала "
            f"«{name}». Обученные функционалы: {', '.join(known)}. "
            "Это не приближение, а недоступный метод (§54 ТЗ)."
        )
        raise ValueError(msg)
    return name


def _check_elements(molecule: Molecule) -> None:
    """Отклоняет элементы вне области применения модели (списком, с символами)."""
    outside = {atom.z for atom in molecule.atoms} - set(dd.D3_ELEMENTS)
    if outside:
        symbols = ", ".join(atom.symbol for atom in molecule.atoms if atom.z in outside)
        msg = (
            f"Элемент(ы) {symbols} вне области применения DFT-D3. "
            f"Модель обучена для: {', '.join(dd.ELEMENT_SYMBOLS)}. "
            "Результат без поправки выдавать не будем (§54 ТЗ)."
        )
        raise ValueError(msg)


def _coordination_numbers(zs: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Дробное координационное число каждого атома.

    Степная экспоненциальная счётная функция s-dftd3:
    ``σ(r) = 1 / (1 + exp(−kcn·(rc/r − 1)))``, rcov — сумма ковалентных
    радиусов пары (Pyykko 2009 × 4/3, в bohr), kcn = 16.
    """
    n = zs.shape[0]
    cns = np.zeros(n)
    rcov = np.array([dd.element_data(int(z)).rcov_bohr for z in zs])
    for i in range(n):
        for j in range(i + 1, n):
            vec = pos[i] - pos[j]
            r2 = float(vec @ vec)
            if r2 > _COORDINATION_CUTOFF_SQ or r2 < _MIN_DISTANCE_SQ:
                continue
            r = np.sqrt(r2)
            countf = 1.0 / (1.0 + np.exp(-dd.KCN * ((rcov[i] + rcov[j]) / r - 1.0)))
            cns[i] += countf
            cns[j] += countf
    return cns


def _cn_weights(cns: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Веса CN-интерполяции и их производные по CN.

    ``w_a ∝ exp(−wf·(CN − c_a)²)``, нормированы на 1; производная
    ``∂w_a/∂CN = (d_a·S − e_a·Σd_b)/S²`` (s-dftd3, ``weight_references``).

    Фолбэк при вырождении весов (все экспоненты подплывают в 0 и 0·∞ = NaN):
    единичный вес на системе отсчёта с **наибольшим** табличным CN для этого
    элемента, производные — ноль. Это дословная семантика s-dftd3
    (``is_exceptional`` + ``maxval``), а не приближение.

    Возвращает массивы формы ``(n_atoms, 7)`` — по максимуму числа отсчётов;
    для элементов с меньшим числом отсчёты нулевые и не влияют на сумму
    (их веса нулевые).
    """
    n = cns.shape[0]
    weights = np.zeros((n, 7))
    dw = np.zeros((n, 7))
    for i in range(n):
        z = int(zs[i])
        el = dd.element_data(z)
        refcn = np.asarray(el.refcn)
        k = el.nref
        cn = cns[i]
        expw = np.exp(-dd.WF * (cn - refcn) ** 2)
        s = float(expw.sum())
        norm = 1.0 / s
        d = 2.0 * dd.WF * (refcn - cn) * expw  # d_a = 2·wf·(c_a − CN)·e_a
        dsum = float(d.sum())
        max_refcn = float(refcn[:k].max())
        for a in range(k):
            gwk = expw[a] * norm
            if not np.isfinite(gwk):
                gwk = 1.0 if refcn[a] == max_refcn else 0.0
            weights[i, a] = gwk
            dgwk = d[a] * norm - expw[a] * dsum * norm * norm
            if not np.isfinite(dgwk):
                dgwk = 0.0
            dw[i, a] = dgwk
    return weights, dw


def _atomic_c6(
    zs: np.ndarray, weights: np.ndarray, dw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CN-интерполированные C6 для всех пар и их производные по CN.

    ``C6_ij = Σ_{a,b} w_{ia}·w_{jb}·C6^{(a,b)}_{z_i z_j}``; производные —
    по одной стороне веса (``dc6dcn[i, j]`` — по CN атома i).
    """
    n = zs.shape[0]
    c6 = np.zeros((n, n))
    dc6dcn = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            zi, zj = int(zs[i]), int(zs[j])
            # Атом i владеет индексом ahi, только если его Z больше.
            i_is_hi = zi > zj
            lo, hi = (zi, zj) if not i_is_hi else (zj, zi)
            # В данных: a_hi — индекс отсчёта элемента с БОЛЬШИМ Z (медленное
            # измерение), a_lo — с малым (быстрое).
            n_lo, n_hi = dd.nref_of(lo), dd.nref_of(hi)
            c = dci = dcj = 0.0
            for ahi in range(n_hi):
                for alo in range(n_lo):
                    refc6 = dd.c6_lookup(lo, hi, ahi, alo)
                    w_i = weights[i, ahi] if i_is_hi else weights[i, alo]
                    w_j = weights[j, alo] if i_is_hi else weights[j, ahi]
                    dw_i = dw[i, ahi] if i_is_hi else dw[i, alo]
                    dw_j = dw[j, alo] if i_is_hi else dw[j, ahi]
                    c += w_i * w_j * refc6
                    dci += dw_i * w_j * refc6
                    dcj += w_i * dw_j * refc6
            c6[i, j] = c6[j, i] = c
            dc6dcn[i, j] = dci
            dc6dcn[j, i] = dcj
    return c6, dc6dcn


def _bj_damping(
    zs: np.ndarray,
    pos: np.ndarray,
    c6: np.ndarray,
    dc6dcn: np.ndarray,
    s8: float,
    a1: float,
    a2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """BJ-затухание: энергия, градиент и ∂E/CN каждого атома.

    r₀ = a1·√rrij + a2, t6 = 1/(r⁶+r₀⁶), t8 = 1/(r⁸+r₀⁸),
    E = −C6·(s6·t6 + s8·rrij·t8) — рациональная форма без сингулярности
    в r = 0 (s6 = 1 у всех обученных параметров).
    """
    n = zs.shape[0]
    r4 = np.array([dd.element_data(int(z)).r4eff for z in zs])
    energy = 0.0
    gradient = np.zeros((n, 3))
    dedcn = np.zeros(n)
    for i in range(n):
        for j in range(i):
            vec = pos[i] - pos[j]
            r2 = float(vec @ vec)
            if r2 > _DISPERSION_CUTOFF_SQ or r2 < _MIN_DISTANCE_SQ:
                continue
            rrij = 3.0 * r4[i] * r4[j]
            r0 = a1 * np.sqrt(rrij) + a2
            t6 = 1.0 / (r2**3 + r0**6)
            t8 = 1.0 / (r2**4 + r0**8)
            edisp = t6 + s8 * rrij * t8  # s6 = 1
            energy -= c6[i, j] * edisp
            d6 = -6.0 * r2**2 * t6**2  # d/dr2 (1/(r2³+r0⁶))
            d8 = -8.0 * r2**3 * t8**2
            gdisp = d6 + s8 * rrij * d8
            delta_force = -c6[i, j] * gdisp * vec
            gradient[i] += delta_force
            gradient[j] -= delta_force
            dedcn[i] -= dc6dcn[i, j] * edisp
            dedcn[j] -= dc6dcn[j, i] * edisp
    return energy, gradient, dedcn


def _zero_damping(
    zs: np.ndarray,
    pos: np.ndarray,
    c6: np.ndarray,
    dc6dcn: np.ndarray,
    s8: float,
    rs6: float,
    rs8: float,
    alp: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Zero-затухание: энергия, градиент и ∂E/CN каждого атома.

    r₀ = вандерваальсов радиус **пары** (не среднего по атомам),
    t6 = (rs6·r0/r)^14, t8 = (rs8·r0/r)^16, f = 1/(1+6t),
    E = −C6·(s6·f6/r⁶ + s8·rrij·f8/r⁸).
    """
    n = zs.shape[0]
    r4 = np.array([dd.element_data(int(z)).r4eff for z in zs])
    alp8 = alp + 2.0
    energy = 0.0
    gradient = np.zeros((n, 3))
    dedcn = np.zeros(n)
    for i in range(n):
        for j in range(i):
            vec = pos[i] - pos[j]
            r2 = float(vec @ vec)
            if r2 > _DISPERSION_CUTOFF_SQ or r2 < _MIN_DISTANCE_SQ:
                continue
            r = np.sqrt(r2)
            rrij = 3.0 * r4[i] * r4[j]
            r0 = dd.vdw_pair(int(zs[i]), int(zs[j]))
            r6 = r2**3
            r8 = r6 * r2
            t6 = (rs6 * r0 / r) ** alp
            t8 = (rs8 * r0 / r) ** alp8
            f6 = 1.0 / (1.0 + 6.0 * t6)
            f8 = 1.0 / (1.0 + 6.0 * t8)
            edisp = f6 / r6 + s8 * rrij * f8 / r8  # s6 = 1
            energy -= c6[i, j] * edisp
            # d/dr2 (f6/r⁶) = (−6·f6 + 6·alp·t6·f6²)/r2 / r⁶
            d6 = (-6.0 * f6 + 6.0 * alp * t6 * f6**2) / r2
            d8 = (-8.0 * f8 + 6.0 * alp8 * t8 * f8**2) / r2
            gdisp = d6 / r6 + s8 * rrij * d8 / r8
            delta_force = -c6[i, j] * gdisp * vec
            gradient[i] += delta_force
            gradient[j] -= delta_force
            dedcn[i] -= dc6dcn[i, j] * edisp
            dedcn[j] -= dc6dcn[j, i] * edisp
    return energy, gradient, dedcn


def _coordination_gradient(
    zs: np.ndarray,
    pos: np.ndarray,
    dedcn: np.ndarray,
) -> np.ndarray:
    """Перенос ∂E/CN в градиент по координатам (цепное правило).

    dσ/dr = −kcn·rc·e^{−x} / (r·(1+e^{−x})²), x = kcn·(rc/r − 1);
    вклад пары (i, j) в G_i: (dEdCN_i + dEdCN_j)·(dσ/drij)·(R_i − R_j)/r.
    """
    n = zs.shape[0]
    gradient = np.zeros((n, 3))
    rcov = np.array([dd.element_data(int(z)).rcov_bohr for z in zs])
    for i in range(n):
        for j in range(i):
            vec = pos[i] - pos[j]
            r2 = float(vec @ vec)
            if r2 > _COORDINATION_CUTOFF_SQ or r2 < _MIN_DISTANCE_SQ:
                continue
            r = np.sqrt(r2)
            rc = rcov[i] + rcov[j]
            x = dd.KCN * (rc / r - 1.0)
            expterm = np.exp(-x)
            dexp = (-dd.KCN * rc * expterm) / (r2 * (expterm + 1.0) ** 2)
            coef = (dedcn[i] + dedcn[j]) * dexp / r
            delta_force = coef * vec
            gradient[i] += delta_force
            gradient[j] -= delta_force
    return gradient


def _atom_arrays(molecule: Molecule) -> tuple[np.ndarray, np.ndarray]:
    """Атомные номера (int64) и координаты в bohr, форма ``(N, 3)``."""
    zs = np.fromiter((atom.z for atom in molecule.atoms), dtype=np.int64, count=molecule.n_atoms)
    pos = np.array([atom.position for atom in molecule.atoms], dtype=np.float64) * angstrom_to_bohr(
        1.0
    )
    return zs, pos


def _c6_for_molecule(molecule: Molecule) -> tuple[np.ndarray, np.ndarray]:
    """CN-интерполированные C6 и их производные по CN для всех пар."""
    zs, pos = _atom_arrays(molecule)
    cns = _coordination_numbers(zs, pos)
    weights, dw = _cn_weights(cns, zs)
    return _atomic_c6(zs, weights, dw)


def dftd3_contribution(
    molecule: Molecule,
    correction: DispersionCorrection,
    functional: str | None,
) -> DispersionContribution:
    """Энергия и аналитический градиент поправки DFT-D3 для молекулы.

    ``functional`` — обменно-корреляционный функционал (``None`` для HF):
    параметры затухания обучаются на конкретный функционал, поэтому без
    него результат был бы другим числом под тем же именем (§54 ТЗ).

    Отклоняет (ValueError до начала расчёта): функционал без обученных
    параметров (включая LDA — для неё параметров не существует) и элементы
    вне области применения модели.
    """
    name = _functional_name(correction, functional)
    _check_elements(molecule)
    zs, pos = _atom_arrays(molecule)
    c6, dc6dcn = _c6_for_molecule(molecule)

    if correction is DispersionCorrection.D3_BJ:
        _s6, s8, a1, a2 = dd.BJ_PARAMS[name]
        del _s6  # s6 = 1 у всех обученных параметров; зафиксировано в данных
        energy, gradient, dedcn = _bj_damping(zs, pos, c6, dc6dcn, s8, a1, a2)
        model = "d3bj"
    elif correction is DispersionCorrection.D3_ZERO:
        _s6, s8, rs6, rs8, alp = dd.ZERO_PARAMS[name]
        del _s6
        energy, gradient, dedcn = _zero_damping(zs, pos, c6, dc6dcn, s8, rs6, rs8, alp)
        model = "d3zero"
    else:
        msg = f"D3 не определён для поправки {correction!r} (есть только d3bj/d3zero)."
        raise ValueError(msg)

    gradient = gradient + _coordination_gradient(zs, pos, dedcn)
    return DispersionContribution(
        model=model,
        functional=name,
        energy_hartree=energy,
        gradient=gradient,
    )
