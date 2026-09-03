"""Дисперсионная поправка DFT-D4: ядро модели (энергия).

Чистая Python-реализация модели D4 (Grimme et al., J. Chem. Theory Comput.
2021, 17, 6579): C6-коэффициенты через справочные системы + BJ-затухание.
Данные — из ``dispersion_d4_data`` (извлечены из dftd4; см. его docstring).

Структура модели (идентична исходнику dftd4, src/dftd4):

* Справочный поляризационный профиль каждой системы (23 точки по частоте):
  ``alpha = max(ascale*(alphaiw − hcount·sscale·secaiw·zeta(ga, η_ref·gc,
  zeff_ref, clsh+zeff_ref)), 0)`` (set_refalpha_eeq).
* Справочный C6 пары систем: ``C6_ref = (3/π)·trapzd(alpha_i ⊙ alpha_j)`` —
  численная интеграция Казимира — Полдера по 23-точечной сетке частот.
* Вес справочной системы на атоме: нормированная гауссова интерполяция по
  координационному числу (``weight_cn`` с ngw гауссианами ширины igw·wf) ×
  зарядовое масштабирование ``zeta`` (set_refgw, weight_references).
* C6 пары атомов: ``Σ_{ab} gw_i(a)·gw_j(b)·C6_ref(a, b)``.
* Энергия (BJ, get_dispersion_energy):
  ``E = −Σ_{i<j} C6_ij·(s6/(r⁶+r0⁶) + s8·⟨3·R4R2⟩/(r⁸+r0⁸))``,
  ``r0 = a1·√(3·R4R2_i·R4R2_j) + a2``, ``R4R2(Z) = √(0.5·r4_over_r2·√Z)``.
  (В dftd4 dE=−½·C6·edisp складывается по обоим атомам пары; полная парная
  энергия — без ½.)
* Координационное число (mctc-lib ``erf_dftd4``-счётчик):
  ``CN_i = Σ_{j≠i} f_en(i,j)·½(1 + erf(−kcn·(r_ij − rc_ij)/rc_ij))``, kcn = 7.5,
  ``rc_ij = (4/3)·(rcov_i + rcov_j)`` (ковалентные радиусы 2009 × 4/3, как в
  dftd4 ``covrad``), ``f_en = 4.10451·exp(−(|χ_i−χ_j|+19.08857)²/(2·11.28174²))``
  с электронегативностями Полинга; пары с r > 30 bohr не учитываются.

Заряды: по умолчанию — EEQ (eeq2019, multicharge) — та же модель, что в
dftd4 по умолчанию (d4_qmod%eeq); см. ``eeq_charges``. Для гомоядерных
систем EEQ даёт точный 0 по симметрии (совпадение со стадией 2).
Явные заряды можно передать ``charges`` (перекрывает EEQ).

Трёхчленный C9-член (s9, Axilrod–Teller угол + CHG-нулевое затухание,
alp=16) считается по C6 при q=0 (как в dftd4 get_dispersion3); на
диатомиках он тождественно нуль. Аналитический градиент — полная сборка
dftd4 get_dispersion: явная геометрия + dedcn·∇ЧК + dedq·(dq/dR) (CP-
решение EEQ; зарядная связь — только парная часть, C9 определён при q=0).

Оракул сверки — libdftd4 (pyscf.dispersion.dftd4, ADR-002), включая
``dftd4_get_dispersion`` с градиентом и ``dftd4_get_properties``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quantumlab.engine import dispersion_d4_data as d4d
from quantumlab.engine.eeq_charges import EEQ2019

# --- Константы модели (значения из исходника dftd4) ---
#: Максимальная высота зарядового масштабирования (ga_default).
GA_DEFAULT = 3.0
#: Крутизна зарядового масштабирования (gc_default).
GC_DEFAULT = 2.0
#: Коэффициент весовой функции ЧК (wf_default).
WF_DEFAULT = 6.0
#: Крутизна erf-счётной функции ЧК (dftd4_ncoord::default_kcn).
KCN = 7.5
#: Масштаб ковалентных радиусов для ЧК: (4/3)·covalent_rad_2009
#: (dftd4 covrad: get_covalent_rad возвращает covalent_rad_d3).
CN_RADIUS_SCALE = 4.0 / 3.0
#: Реальное пространство-отсечение ЧК (dftd4 cutoff::cn_default, bohr).
CN_CUTOFF = 30.0
#: EN-масштабирование erf-счётчика (mctc-lib erf_dftd4: k4/k5/k6).
EN_K4 = 4.10451
EN_K5 = 19.08857
EN_K6 = 2.0 * 11.28174**2
#: 3/π — множитель C–P интеграции (thopi).
THOPI = 3.0 / math.pi
#: Пространственное отсечение парной дисперсии, bohr (dftd4 disp2_default).
DISP2_CUTOFF = 60.0
#: Пространственное отсечение трёхчленного C9-члена, bohr (dftd4 disp3_default).
DISP3_CUTOFF = 40.0
#: Порог r² против нуля (Fortran: epsilon(1.0)).
_EPS_R2 = np.finfo(float).eps
#: 23-точечная сетка частот интеграции Казимира — Полдера (trapzd).
CP_FREQ: tuple[float, ...] = (
    1e-6,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
)
#: Порог нормы весов (Fortran: tiny(1.0)**0.5).
_EPS_NORM = math.sqrt(np.finfo(float).tiny)
#: Неподдерживаемые dftd4 элементы (Rf–Rg) — отклоняются как в исходнике.
_UNSUPPORTED_Z = frozenset(range(104, 112))


def _trapzd_weights(freq: Sequence[float]) -> list[float]:
    """Трапециевые веса на неравномерной сетке (dftd4: weights = 0.5*(…))."""
    n = len(freq)
    out = [0.0] * n
    for i in range(n):
        left = freq[i] - freq[i - 1] if i > 0 else 0.0
        right = freq[i + 1] - freq[i] if i < n - 1 else 0.0
        out[i] = 0.5 * (left + right)
    return out


_CP_WEIGHTS: tuple[float, ...] = tuple(_trapzd_weights(CP_FREQ))


def weight_cn(wf: float, cn: float, cnref: float) -> float:
    """Гауссов вес интерполяции по ЧК: exp(−wf·(cn−cnref)²) (weight_cn)."""
    return math.exp(-wf * (cn - cnref) ** 2)


def zeta(a: float, c: float, qref: float, qmod: float) -> float:
    """Зарядовое масштабирование (zeta); qmod<0 → exp(a) (ветвь изолята)."""
    if qmod < 0.0:
        return math.exp(a)
    return math.exp(a * (1.0 - math.exp(c * (1.0 - qref / qmod))))


def dzeta(a: float, c: float, qref: float, qmod: float) -> float:
    """∂zeta/∂qmod (dftd4 utils::dzeta); qmod<0 → 0 (ветвь изолята)."""
    if qmod < 0.0:
        return 0.0
    return -a * c * math.exp(c * (1.0 - qref / qmod)) * zeta(a, c, qref, qmod) * qref / qmod**2


def trapzd(pol: Sequence[float]) -> float:
    """Численная C–P интеграция по 23-точечной сетке: Σ w_i·pol_i."""
    return sum(w * p for w, p in zip(_CP_WEIGHTS, pol, strict=True))


def cn_pair(r: float, rc: float, kcn: float = KCN) -> float:
    """Вклад пары в ЧК: ½(1+erf(−kcn·(r−rc)/rc)) (mctc-lib erf-счётчик)."""
    return 0.5 * (1.0 + math.erf(-kcn * (r - rc) / rc))


def cn_en_factor(zen_i: float, zen_j: float) -> float:
    """EN-фактор erf-счётчика (mctc-lib erf_dftd4): k4·exp(−(|Δχ|+k5)²/k6)."""
    return EN_K4 * math.exp(-((abs(zen_i - zen_j) + EN_K5) ** 2) / EN_K6)


@dataclass(frozen=True)
class _RefElement:
    """Предвычисленные поэлементные данные одной справочной модели D4."""

    z: int
    nref: int
    #: Справочные ЧК (refcovcn), порядок ir = 1..nref.
    cnref: tuple[float, ...]
    #: Справочные заряды (clsq).
    clsq: tuple[float, ...]
    #: Число гауссиан на систему (set_refgw: треугольное число класса ЧК).
    ngw: tuple[int, ...]
    #: Эффективные поляризационные профили (set_refalpha_eeq), 23 точки.
    alpha: tuple[tuple[float, ...], ...]
    zeff: float
    eta: float
    #: Электронегативность Полинга (для EN-фактора ЧК).
    en: float
    #: R4R2(Z) = sqrt(0.5·r4_over_r2·sqrt(Z)) — в а.е. (bohr²).
    r4r2: float
    #: Ковалентный радиус для ЧК в bohr: (4/3)·covalent_rad_2009.
    rcov_bohr: float


def _nint(x: float) -> int:
    """Fortran nint: округление к ближайшему, полуцелые — от нуля."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def _element_data(z: int, ga: float, gc: float) -> _RefElement:
    """Собирает поэлементные данные (профили, веса, константы) для Z."""
    nref = int(d4d.D4_NREF[z - 1])
    refcn = tuple(float(d4d.D4_REF_CN.get((ir, z), 0.0)) for ir in range(1, nref + 1))

    # set_refgw: cnc(0)=1; счётчик систем с одинаковым nint(refcn) (≤19);
    # ngw(ir) = cnc(класс)·(cnc(класс)+1)/2.
    classes = tuple(min(_nint(rc), 19) for rc in refcn)
    cnc: dict[int, int] = {0: 1}
    for c in classes:
        cnc[c] = cnc.get(c, 0) + 1
    ngw = tuple(cnc[c] * (cnc[c] + 1) // 2 for c in classes)

    cnref = tuple(float(d4d.D4_REF_COV_CN.get((ir, z), 0.0)) for ir in range(1, nref + 1))
    clsq = tuple(float(d4d.D4_CLS_Q.get((ir, z), 0.0)) for ir in range(1, nref + 1))

    # set_refalpha_eeq: эффективный профиль по каждой справочной системе.
    alphas: list[tuple[float, ...]] = []
    for ir in range(1, nref + 1):
        key = (ir, z)
        is_ = int(d4d.D4_REF_SYS.get(key, 0))
        if is_ == 0:  # пустой слот (refsys=0) — вклад 0, как в dftd4 (cycle)
            alphas.append((0.0,) * 23)
            continue
        iz = float(d4d.ZEFF[is_ - 1])
        eta_ref = float(d4d.CHEMICAL_HARDNESS[is_ - 1])
        sscale = float(d4d.REF_SYSSCALE[is_])
        secaiw = d4d.REF_SYSAIW[is_]
        clsh = float(d4d.D4_CLS_H.get(key, 0.0))
        zfac = zeta(ga, eta_ref * gc, iz, clsh + iz)
        ascale = float(d4d.D4_ASCALE.get(key, 0.0))
        hcount = float(d4d.D4_HCOUNT.get(key, 0.0))
        aiw = d4d.D4_ALPHA_IW.get(key, (0.0,) * 23)
        alphas.append(
            tuple(
                max(ascale * (ai - hcount * sscale * se * zfac), 0.0)
                for ai, se in zip(aiw, secaiw, strict=True)
            )
        )

    return _RefElement(
        z=z,
        nref=nref,
        cnref=cnref,
        clsq=clsq,
        ngw=ngw,
        alpha=tuple(alphas),
        zeff=float(d4d.ZEFF[z - 1]),
        eta=float(d4d.CHEMICAL_HARDNESS[z - 1]),
        r4r2=math.sqrt(0.5 * float(d4d.R4_OVER_R2[z - 1]) * math.sqrt(z)),
        # Радиус для ЧК: (4/3)·covalent_rad_2009 (dftd4 covrad::covalent_rad_d3).
        rcov_bohr=CN_RADIUS_SCALE * float(d4d.COVALENT_RAD_ANGSTROM[z - 1]) * d4d.AATOA,
        en=float(d4d.PAULING_EN[z - 1]),
    )


@dataclass(frozen=True)
class D4Result:
    """Результат расчёта D4: энергия (э.е.), ЧК, C6-матрица и заряды."""

    energy_hartree: float
    cn: tuple[float, ...]
    c6: tuple[tuple[float, ...], ...]
    charges: tuple[float, ...]


class DFTD4:
    """Ядро модели D4: C6-коэффициенты и дисперсионная энергия (BJ).

    Аргументы:
        zs: атомные номера.
        pos_bohr: координаты (nat, 3) в bohr.
        xc: функционал — определяет параметры BJ-затухания (bj-eeq-atm).
        charges: частичные заряды; ``None`` (по умолчанию) — модель EEQ
            (eeq2019, дефолт dftd4), явно заданные — используются как есть.
        total_charge: полный заряд системы для EEQ (по умолчанию 0).
        ga/gc/wf: параметры модели (дефолты dftd4: 3.0/2.0/6.0).

    Отклоняет (ValueError): функционал без обученных параметров, элементы
    вне 1..118, неподдерживаемые dftd4 (Rf–Rg) и Z>103 при расчёте EEQ
    (таблицы eeq2019 до Np).
    """

    def __init__(
        self,
        zs: Sequence[int] | np.ndarray,
        pos_bohr: np.ndarray,
        xc: str = "pbe",
        charges: Sequence[float] | None = None,
        total_charge: float = 0.0,
        ga: float = GA_DEFAULT,
        gc: float = GC_DEFAULT,
        wf: float = WF_DEFAULT,
    ) -> None:
        """Инициализация: проверка входных данных, параметры затухания."""
        self.zs = [int(z) for z in zs]
        for z in self.zs:
            if not 1 <= z <= 118:
                msg = f"D4: атомный номер {z} вне области 1..118"
                raise ValueError(msg)
            if z in _UNSUPPORTED_Z:
                msg = f"D4: элемент Z={z} не поддерживается моделью (Rf–Rg)"
                raise ValueError(msg)
        self.pos = np.asarray(pos_bohr, dtype=float)
        if self.pos.ndim != 2 or self.pos.shape[0] != len(self.zs):
            msg = f"DFTD4: размерность pos {self.pos.shape} не совпадает с nat={len(self.zs)}"
            raise ValueError(msg)
        self.nat = len(self.zs)
        self._eeq: EEQ2019 | None = None
        if charges is not None:
            if len(charges) != self.nat:
                msg = f"DFTD4: задано {len(charges)} зарядов при nat={self.nat}"
                raise ValueError(msg)
            self.q = [float(q) for q in charges]
        else:
            # Дефолт dftd4: модель зарядов EEQ (eeq2019).
            self._eeq = EEQ2019(self.zs, self.pos, total_charge)
            self.q = [float(q) for q in self._eeq.charges()]
        self.ga, self.gc, self.wf = ga, gc, wf

        xc_lc = xc.lower()
        params = d4d.D4_DAMPING_PARAMS.get(xc_lc)
        if params is None:
            msg = (
                f"D4: нет обученных параметров для функционала {xc!r} "
                f"(есть {len(d4d.D4_DAMPING_PARAMS)}; см. D4_DAMPING_PARAMS)"
            )
            raise ValueError(msg)
        # Порядок в D4_DAMPING_PARAMS: (s6, s8, a1, a2, s9, alp).
        self.s6, self.s8, self.a1, self.a2, self.s9, self.alp = (float(v) for v in params)

        self.elements = [_element_data(z, ga, gc) for z in self.zs]
        self._refc6 = self._precompute_refc6()

    def _precompute_refc6(self) -> dict[tuple[int, int], np.ndarray]:
        """C6_ref для всех пар элементов молекулы: thopi·trapzd(α_a ⊙ α_b)."""
        cache: dict[tuple[int, int], np.ndarray] = {}
        zset = sorted(set(self.zs))
        data = {z: _element_data(z, self.ga, self.gc) for z in zset}
        for za in zset:
            for zb in zset:
                if za > zb:
                    continue
                ea, eb = data[za], data[zb]
                mat = np.zeros((ea.nref, eb.nref))
                for a in range(ea.nref):
                    for b in range(eb.nref):
                        prod = [pa * pb for pa, pb in zip(ea.alpha[a], eb.alpha[b], strict=True)]
                        mat[a, b] = THOPI * trapzd(prod)
                cache[(za, zb)] = mat
                cache[(zb, za)] = mat.T
        return cache

    def coordination_numbers(self) -> np.ndarray:
        """Дробные ЧК (mctc-lib erf_dftd4-счётчик, kcn=7.5).

        ``CN_i = Σ_{j≠i} f_en(i,j)·½(1+erf(−kcn·(r_ij−rc_ij)/rc_ij))``, где
        ``rc_ij = (4/3)·(rcov_i+rcov_j)`` (bohr), пары с r > 30 bohr не
        учитываются (dftd4 cutoff::cn_default).
        """
        n = self.nat
        cn = np.zeros(n)
        for i in range(n):
            el_i = self.elements[i]
            for j in range(n):
                if i == j:
                    continue
                r = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if r > CN_CUTOFF:
                    continue
                rc = el_i.rcov_bohr + self.elements[j].rcov_bohr
                cn[i] += cn_en_factor(el_i.en, self.elements[j].en) * cn_pair(r, rc)
        return cn

    def _ref_weights(self, cn: np.ndarray, q: np.ndarray | None = None) -> list[list[float]]:
        """Веса справочных систем gwvec (weight_references, q — по атомам)."""
        if q is None:
            q = np.asarray(self.q)
        out: list[list[float]] = []
        for i in range(self.nat):
            el = self.elements[i]
            gi = el.eta * self.gc
            zmod = q[i] + el.zeff
            norm = 0.0
            for a in range(el.nref):
                for igw in range(1, el.ngw[a] + 1):
                    norm += weight_cn(igw * self.wf, cn[i], el.cnref[a])
            inv_norm = 1.0 / norm if abs(norm) > _EPS_NORM else 0.0
            vec: list[float] = []
            for a in range(el.nref):
                expw = 0.0
                for igw in range(1, el.ngw[a] + 1):
                    expw += weight_cn(igw * self.wf, cn[i], el.cnref[a])
                gwk = expw * inv_norm
                if math.isnan(gwk) or math.isinf(gwk) or inv_norm == 0.0:
                    maxcn = max(el.cnref)
                    gwk = 1.0 if abs(maxcn - el.cnref[a]) < 1e-12 else 0.0
                vec.append(gwk * zeta(self.ga, gi, el.clsq[a] + el.zeff, zmod))
            out.append(vec)
        return out

    def _ref_weights_derivs(
        self, cn: np.ndarray, q: np.ndarray
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Векторы gwvec и производные weight_references по ЧК и заряду.

        Возвращает ``(gw, gwdcn, gwdq)`` — списки строк (nat × nref_a;
        длина строк различается). Формулы — из dftd4 model/d4.f90 (гауссов
        ``weight_cn``: d/dcn = 2·wf·(cnref−cn)·w).
        """
        # nref различается по элементам — собираем списком и потом в np
        gws: list[list[float]] = []
        gwdcns: list[list[float]] = []
        gwdqs: list[list[float]] = []
        for i in range(self.nat):
            el = self.elements[i]
            gi = el.eta * self.gc
            zmod = q[i] + el.zeff
            norm = 0.0
            dnorm = 0.0
            for a in range(el.nref):
                for igw in range(1, el.ngw[a] + 1):
                    w = weight_cn(igw * self.wf, cn[i], el.cnref[a])
                    norm += w
                    dnorm += 2.0 * (igw * self.wf) * (el.cnref[a] - cn[i]) * w
            inv_norm = 1.0 / norm if abs(norm) > _EPS_NORM else 0.0
            vec = [0.0] * el.nref
            dvec = [0.0] * el.nref
            qvec = [0.0] * el.nref
            for a in range(el.nref):
                expw = 0.0
                expd = 0.0
                for igw in range(1, el.ngw[a] + 1):
                    w = weight_cn(igw * self.wf, cn[i], el.cnref[a])
                    expw += w
                    expd += 2.0 * (igw * self.wf) * (el.cnref[a] - cn[i]) * w
                gwk = expw * inv_norm
                if math.isnan(gwk) or math.isinf(gwk) or inv_norm == 0.0:
                    maxcn = max(el.cnref)
                    gwk = 1.0 if abs(maxcn - el.cnref[a]) < 1e-12 else 0.0
                zref = el.clsq[a] + el.zeff
                vec[a] = gwk * zeta(self.ga, gi, zref, zmod)
                qvec[a] = gwk * dzeta(self.ga, gi, zref, zmod)
                dgwk = inv_norm * (expd - expw * dnorm * inv_norm)
                if math.isnan(dgwk) or math.isinf(dgwk) or inv_norm == 0.0:
                    dgwk = 0.0
                dvec[a] = dgwk * zeta(self.ga, gi, zref, zmod)
            gws.append(vec)
            gwdcns.append(dvec)
            gwdqs.append(qvec)
        return gws, gwdcns, gwdqs

    def _c6_with_derivs(
        self,
        gw: list[list[float]],
        gwdcn: list[list[float]],
        gwdq: list[list[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """C6 и производные по ЧК/заряду (dftd4 get_atomic_c6)."""
        n = self.nat
        c6 = np.zeros((n, n))
        dc6dcn = np.zeros((n, n))
        dc6dq = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ref = self._refc6[(self.zs[i], self.zs[j])]
                row_i = np.asarray(gw[i])
                row_j = np.asarray(gw[j])
                c6[i, j] = float(np.sum(row_i[:, None] * row_j[None, :] * ref))
                c6[j, i] = c6[i, j]
                dc6dcn[i, j] = float(np.sum(np.asarray(gwdcn[i])[:, None] * row_j[None, :] * ref))
                dc6dcn[j, i] = float(np.sum(row_i[:, None] * np.asarray(gwdcn[j])[None, :] * ref))
                dc6dq[i, j] = float(np.sum(np.asarray(gwdq[i])[:, None] * row_j[None, :] * ref))
                dc6dq[j, i] = float(np.sum(row_i[:, None] * np.asarray(gwdq[j])[None, :] * ref))
        return c6, dc6dcn, dc6dq

    def _c6_matrix_from_weights(self, gw: list[list[float]]) -> np.ndarray:
        """C6 из готовых весов справочных систем (без повторного расчёта)."""
        n = self.nat
        c6 = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ref = self._refc6[(self.zs[i], self.zs[j])]
                row_i = np.asarray(gw[i])
                row_j = np.asarray(gw[j])
                c6[i, j] = float(np.sum(row_i[:, None] * row_j[None, :] * ref))
                c6[j, i] = c6[i, j]
        return c6

    def c6_matrix(self, cn: np.ndarray | None = None) -> np.ndarray:
        """C6-коэффициенты всех пар атомов (ат. е.)."""
        if cn is None:
            cn = self.coordination_numbers()
        return self._c6_matrix_from_weights(self._ref_weights(cn))

    def _pair_terms(
        self, c6: np.ndarray
    ) -> tuple[float, dict[tuple[int, int], tuple[float, float, float, float]]]:
        """Парный BJ-вклад по всем парам.

        Возвращает ``(e_pair, pairs)``; ``pairs[(i, j)] = (edisp0, r, c6ij,
        gdisp0)`` — для градиента (только пары в пределах отсечения 60 bohr).
        """
        n = self.nat
        total = 0.0
        pairs: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        for i in range(n):
            ei = self.elements[i]
            for j in range(i + 1, n):
                ej = self.elements[j]
                vec = self.pos[i] - self.pos[j]
                r2 = float(vec @ vec)
                if r2 < _EPS_R2 or r2 > DISP2_CUTOFF**2:
                    continue
                r = math.sqrt(r2)
                rj = 3.0 * ei.r4r2 * ej.r4r2
                r0 = self.a1 * math.sqrt(rj) + self.a2
                t6 = 1.0 / (r2**3 + r0**6)
                t8 = 1.0 / (r2**4 + r0**8)
                edisp0 = self.s6 * t6 + self.s8 * rj * t8
                # d6/d8 — 2·d/d(r²) из dftd4 get_dispersion_derivs
                d6 = -6.0 * r2**2 * t6**2
                d8 = -8.0 * r2**3 * t8**2
                gdisp0 = self.s6 * d6 + self.s8 * rj * d8
                c6ij = c6[i, j]
                # dftd4 дробит dE=−½·C6·edisp по двум атомам; полная парная
                # энергия (сумма по атомам) = −C6·edisp. Итерация по i<j —
                # по одной паре, множитель ½ уже учтён разбиением.
                total += -c6ij * edisp0
                pairs[(i, j)] = (edisp0, r, c6ij, gdisp0)
        return total, pairs

    def c9_energy(self, c6_q0: np.ndarray) -> float:
        """Трёхчленный C9-член (ATM + CHG-затухание, s9), C6 при q=0.

        dftd4 get_atm_dispersion_energy: по тройкам i>j>k
        ``E_trip = ang·fdmp·c9·triple_scale`` (triple_scale=1 для трёх
        различных атомов), ``c9 = −s9·√|C6ij·C6ik·C6jk|``,
        ``fdmp = 1/(1+6(r0/r1)^(alp/3))``,
        ``ang = 0.375·(r²ij+r²jk−r²ik)(r²ij−r²jk+r²ik)(−r²ij+r²jk+r²ik)/
        (r²ij·r²jk·r²ik)^(5/2) + (r²ij·r²jk·r²ik)^(−3/2)``; в энергию
        входит ``−E_trip`` (c9<0 → вклад обычно положительный), каждая из
        трёх атомных энергий получает ``−E_trip/3``.
        """
        n = self.nat
        total = 0.0
        alp3 = self.alp / 3.0
        for i in range(n):
            ei = self.elements[i]
            for j in range(i):
                ej = self.elements[j]
                for k in range(j):
                    ek = self.elements[k]
                    vij = self.pos[j] - self.pos[i]
                    vik = self.pos[k] - self.pos[i]
                    vjk = self.pos[k] - self.pos[j]
                    r2ij = float(vij @ vij)
                    r2ik = float(vik @ vik)
                    r2jk = float(vjk @ vjk)
                    if (
                        r2ij < _EPS_R2
                        or r2ik < _EPS_R2
                        or r2jk < _EPS_R2
                        or r2ij > DISP3_CUTOFF**2
                        or r2ik > DISP3_CUTOFF**2
                        or r2jk > DISP3_CUTOFF**2
                    ):
                        continue
                    c6ij = c6_q0[i, j]
                    c6ik = c6_q0[i, k]
                    c6jk = c6_q0[j, k]
                    c9 = -self.s9 * math.sqrt(abs(c6ij * c6ik * c6jk))
                    r0ij = self.a1 * math.sqrt(3.0 * ej.r4r2 * ei.r4r2) + self.a2
                    r0ik = self.a1 * math.sqrt(3.0 * ek.r4r2 * ei.r4r2) + self.a2
                    r0jk = self.a1 * math.sqrt(3.0 * ek.r4r2 * ej.r4r2) + self.a2
                    r0 = r0ij * r0ik * r0jk
                    p = r2ij * r2ik * r2jk
                    r1 = math.sqrt(p)
                    fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** alp3)
                    ang = (
                        0.375
                        * (r2ij + r2jk - r2ik)
                        * (r2ij - r2jk + r2ik)
                        * (-r2ij + r2jk + r2ik)
                        / p**2.5
                        + p**-1.5
                    )
                    total += -ang * fdmp * c9
        return total

    def energy(self, cn: np.ndarray | None = None, c6: np.ndarray | None = None) -> float:
        """Полная дисперсионная энергия (э.е.): пары (BJ) + C9-трёхчленный.

        C9-член считается по C6 при q=0 (dftd4 обнуляет заряды перед
        get_dispersion3). На диатомиках C9 тождественно нуль.
        """
        if cn is None:
            cn = self.coordination_numbers()
        if c6 is None:
            c6 = self.c6_matrix(cn)
        e_pair, _ = self._pair_terms(c6)
        if self.s9 != 0.0 and self.nat > 2:
            gw0 = self._ref_weights(cn, q=np.zeros(self.nat))
            c6_q0 = self._c6_matrix_from_weights(gw0)
            return e_pair + self.c9_energy(c6_q0)
        return e_pair

    def _c9_triplet_terms(
        self, c6_q0: np.ndarray, dc6_q0: tuple[np.ndarray, np.ndarray]
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """C9-тройки: энергия, вклад в градиент (явная геометрия), dE/dC6.

        ``dc6_q0`` — производные C6 (при q=0) по ЧК: (dc6dcn, dc6dq) — для
        копактуации dedcn/dedq; дробится как в dftd4:
        ``dedcn[i] −= dE·0.5·(dc6dcn[i,j]/C6ij + dc6dcn[i,k]/C6ik)``.
        """
        n = self.nat
        grad = np.zeros((n, 3))
        dedcn = np.zeros(n)
        dedq = np.zeros(n)
        total = 0.0
        alp3 = self.alp / 3.0
        for i in range(n):
            ei = self.elements[i]
            for j in range(i):
                ej = self.elements[j]
                for k in range(j):
                    ek = self.elements[k]
                    vij = self.pos[j] - self.pos[i]
                    vik = self.pos[k] - self.pos[i]
                    vjk = self.pos[k] - self.pos[j]
                    r2ij = float(vij @ vij)
                    r2ik = float(vik @ vik)
                    r2jk = float(vjk @ vjk)
                    if (
                        r2ij < _EPS_R2
                        or r2ik < _EPS_R2
                        or r2jk < _EPS_R2
                        or r2ij > DISP3_CUTOFF**2
                        or r2ik > DISP3_CUTOFF**2
                        or r2jk > DISP3_CUTOFF**2
                    ):
                        continue
                    c6ij = c6_q0[i, j]
                    c6ik = c6_q0[i, k]
                    c6jk = c6_q0[j, k]
                    c9 = -self.s9 * math.sqrt(abs(c6ij * c6ik * c6jk))
                    r0ij = self.a1 * math.sqrt(3.0 * ej.r4r2 * ei.r4r2) + self.a2
                    r0ik = self.a1 * math.sqrt(3.0 * ek.r4r2 * ei.r4r2) + self.a2
                    r0jk = self.a1 * math.sqrt(3.0 * ek.r4r2 * ej.r4r2) + self.a2
                    r0 = r0ij * r0ik * r0jk
                    p = r2ij * r2ik * r2jk
                    r1 = math.sqrt(p)
                    fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** alp3)
                    ang = (
                        0.375
                        * (r2ij + r2jk - r2ik)
                        * (r2ij - r2jk + r2ik)
                        * (-r2ij + r2jk + r2ik)
                        / p**2.5
                        + p**-1.5
                    )
                    de0 = ang * fdmp * c9  # triple_scale=1, sw=1 (молекула)
                    total += -de0
                    # Явная геометрическая часть (dftd4 get_atm_dispersion_derivs).
                    dfdmp = -2.0 * self.alp * (r0 / r1) ** alp3 * fdmp**2
                    dang_ij = (
                        -0.375
                        * (
                            r2ij**3
                            + r2ij**2 * (r2jk + r2ik)
                            + r2ij * (3.0 * r2jk**2 + 2.0 * r2jk * r2ik + 3.0 * r2ik**2)
                            - 5.0 * (r2jk - r2ik) ** 2 * (r2jk + r2ik)
                        )
                        / p**2.5
                    )
                    dang_ik = (
                        -0.375
                        * (
                            r2ik**3
                            + r2ik**2 * (r2jk + r2ij)
                            + r2ik * (3.0 * r2jk**2 + 2.0 * r2jk * r2ij + 3.0 * r2ij**2)
                            - 5.0 * (r2jk - r2ij) ** 2 * (r2jk + r2ij)
                        )
                        / p**2.5
                    )
                    dang_jk = (
                        -0.375
                        * (
                            r2jk**3
                            + r2jk**2 * (r2ik + r2ij)
                            + r2jk * (3.0 * r2ik**2 + 2.0 * r2ik * r2ij + 3.0 * r2ij**2)
                            - 5.0 * (r2ik - r2ij) ** 2 * (r2ik + r2ij)
                        )
                        / p**2.5
                    )
                    dgij = c9 * (-dang_ij * fdmp + ang * dfdmp) / r2ij * vij
                    dgik = c9 * (-dang_ik * fdmp + ang * dfdmp) / r2ik * vik
                    dgjk = c9 * (-dang_jk * fdmp + ang * dfdmp) / r2jk * vjk
                    grad[i] -= dgij + dgik
                    grad[j] += dgij - dgjk
                    grad[k] += dgik + dgjk
                    # Копактуация по C6 (C6(q=0)): ∂E/C6ab = −de0·0.5/C6ab
                    # (E_trip = −de0 ∝ √(C6ij·C6ik·C6jk)).
                    dedcn[i] -= de0 * 0.5 * (dc6_q0[0][i, j] / c6ij + dc6_q0[0][i, k] / c6ik)
                    dedcn[j] -= de0 * 0.5 * (dc6_q0[0][j, i] / c6ij + dc6_q0[0][j, k] / c6jk)
                    dedcn[k] -= de0 * 0.5 * (dc6_q0[0][k, i] / c6ik + dc6_q0[0][k, j] / c6jk)
                    dedq[i] -= de0 * 0.5 * (dc6_q0[1][i, j] / c6ij + dc6_q0[1][i, k] / c6ik)
                    dedq[j] -= de0 * 0.5 * (dc6_q0[1][j, i] / c6ij + dc6_q0[1][j, k] / c6jk)
                    dedq[k] -= de0 * 0.5 * (dc6_q0[1][k, i] / c6ik + dc6_q0[1][k, j] / c6jk)
        return total, grad, dedcn, dedq

    def gradient(self) -> np.ndarray:
        """Аналитический градиент D4 (nat, 3), э.е./боhr.

        Сборка — как в dftd4 get_dispersion:
        1) пары: явная BJ-часть + dedcn/dedq (C6 при фактических зарядах);
        2) grad += dedq·(dq/dR) — только парная часть dedq (в dftd4
           d4_gemv вызывается до C9; C9 определён при q=0, его dedq в
           градиент не входит);
        3) C9 при q=0: явная часть + dedcn/dedq (накапливаются);
        4) grad += dEdcn_total·(∇ЧК C6) — один раз, по суммарному dedcn.

        dq/dR — CP-решение EEQ (``eeq_charges.EEQ2019.dqdr``); для явных
        ``charges`` (не EEQ) dq/dR = 0 (заряды считаются фиксированными).
        """
        n = self.nat
        cn = self.coordination_numbers()
        q = np.array(self.q)
        gw, gwdcn, gwdq = self._ref_weights_derivs(cn, q)
        c6, dc6dcn, dc6dq = self._c6_with_derivs(gw, gwdcn, gwdq)
        grad = np.zeros((n, 3))
        dedcn = np.zeros(n)
        dedq = np.zeros(n)

        # 1) пары: явная геометрия + производные по C6
        _, pairs = self._pair_terms(c6)
        for (i, j), (edisp0, _r, c6ij, gdisp0) in pairs.items():
            vec = self.pos[i] - self.pos[j]
            dg = -c6ij * gdisp0 * vec
            grad[i] += dg
            grad[j] -= dg
            dedcn[i] -= dc6dcn[i, j] * edisp0
            dedcn[j] -= dc6dcn[j, i] * edisp0
            dedq[i] -= dc6dq[i, j] * edisp0
            dedq[j] -= dc6dq[j, i] * edisp0

        # 2) зарядная связь (только парная часть dedq — как в dftd4)
        if self._eeq is not None:
            dqdr = self._eeq.dqdr()  # (a, j, ic): ∂q_j/∂R_a^ic
            grad += np.einsum("aji,j->ai", dqdr, dedq)

        # 3) C9 при q=0
        if self.s9 != 0.0 and n > 2:
            gw0, gwdcn0, gwdq0 = self._ref_weights_derivs(cn, np.zeros(n))
            c6_q0, dc6dcn0, dc6dq0 = self._c6_with_derivs(gw0, gwdcn0, gwdq0)
            _e_c9, g_c9, dedcn_c9, _dedq_c9 = self._c9_triplet_terms(c6_q0, (dc6dcn0, dc6dq0))
            grad += g_c9
            dedcn += dedcn_c9
            # _dedq_c9 накапливается, но в градиент не входит (см. выше).

        # 4) связь по ЧК C6 (mctc add_coordination_number_derivs)
        for i in range(n):
            el_i = self.elements[i]
            for j in range(i + 1, n):
                el_j = self.elements[j]
                vec = self.pos[i] - self.pos[j]
                r2 = float(vec @ vec)
                if r2 < 1e-12 or r2 > CN_CUTOFF**2:
                    continue
                r = math.sqrt(r2)
                rc = el_i.rcov_bohr + el_j.rcov_bohr
                x = KCN * (r - rc) / rc
                dcount = -KCN * math.exp(-(x * x)) / (math.sqrt(math.pi) * rc)
                countd = cn_en_factor(el_i.en, el_j.en) * dcount * vec / r
                grad[i] += countd * (dedcn[i] + dedcn[j])
                grad[j] -= countd * (dedcn[i] + dedcn[j])
        return grad

    def get_dispersion(self) -> D4Result:
        """Полный результат: энергия + ЧК + C6-матрица + заряды (сверка/отладка)."""
        cn = self.coordination_numbers()
        c6 = self.c6_matrix(cn)
        return D4Result(
            energy_hartree=self.energy(cn=cn, c6=c6),
            cn=tuple(float(x) for x in cn),
            c6=tuple(tuple(float(x) for x in row) for row in c6),
            charges=tuple(float(q) for q in self.q),
        )
