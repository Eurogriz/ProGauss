"""Модель зарядов EEQ (eeq2019) — чистый Python-порт для D4.

EEQ (electronegativity equalization) — дефолтная модель частичных зарядов
в dftd4 4.x: ``new_d4_model`` по умолчанию вызывает ``new_eeq2019_model``
(multicharge). Чистый Python-порт по закреплённым исходникам:

* ``tools/multicharge_sources/`` — multicharge main @2026-08-31:
  ``param.f90`` (``new_eeq2019_model``: cutoff=25 bohr, kcn=7.5, cn_max=8.0,
  rcov=mctc::get_covalent_rad), ``model/eeq.f90`` (xvec, матрица A),
  ``model/type.F90`` (``solve``: Bunch-Kaufman, ``local_charge``),
  ``param/eeq2019.f90`` (таблицы; Caldeweyher et al., J. Chem. Phys. 2019,
  150, 154122, DOI 10.1063/1.5090222);
* ``tools/mctc_sources/`` — mctc-lib main @2026-08-31: erf-счётчик ЧК
  (без EN-фактора, в отличие от счётчика C6) и плавное ограничение ЧК.

Модель:
* ЧК зарядовой модели (не путать с ЧК C6 — там cutoff 30 bohr и EN-фактор):
  ``CN_i = Σ_{j≠i} ½(1+erf(−kcn·(r_ij−rc_ij)/rc_ij))``, kcn=7.5,
  ``rc_ij = rcov_i+rcov_j`` с ``rcov = (4/3)·covalent_rad_2009`` (bohr),
  пары с r > 25 bohr не учитываются, затем плавное ограничение
  ``CN = ln(1+e⁸) − ln(1+e^(8−CN))`` (асимптотически к 8, не min);
* линейная система ``A·[q; λ] = x`` размерности (nat+1)×(nat+1):
  ``A_ii = η_i + √(2/π)/rad_i``, ``A_ij = erf(√(r²γ_ij))/r``,
  ``γ_ij = 1/(rad_i²+rad_j²)``, последняя строка/столбец = 1,
  ``A_{n+1,n+1} = 0``; ``x_i = −χ_i + kcnchi_i·CN_i/√(CN_i+1e-14)``,
  ``x_{n+1} = Q`` (полный заряд);
* решение — charges ``q_i`` (λ — множитель Лагранжа ограничения
  ``Σq = Q``; заряды автоматически суммируются в Q).

Таблицы ``D4_EEQ_*`` (103 элемента, Z=1..103) — из
``dispersion_d4_data`` (генератор ``tools/generate_d4_data.py``).
Для Z>103 референсной модели зарядов нет — отклоняется ValueError.
Оракул сверки — libdftd4 4.0.1 (pyscf.dispersion.dftd4, ADR-002).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from quantumlab.engine import dispersion_d4_data as d4d

#: Крутизна erf-счётчика ЧК (new_eeq2019_model: cn_exp).
EEQ_KCN = 7.5
#: Реальное отсечение ЧК зарядовой модели, bohr (new_eeq2019_model: cutoff).
#: Не 30 bohr — это отсечение ЧК модели C6.
EEQ_CN_CUTOFF = 25.0
#: Плавный предел ЧК зарядовой модели (new_eeq2019_model: cn_max).
EEQ_CN_MAX = 8.0
#: Регуляризатор корня в xvec (multicharge eeq.f90: reg).
_REG = 1.0e-14
#: √(2/π) — вклад гауссова заряда на диагонали A (eeq.f90: sqrt2pi).
_SQRT2PI = math.sqrt(2.0 / math.pi)
#: Максимальный Z с таблицами EEQ (multicharge: max_elem = 103).
MAX_Z = len(d4d.D4_EEQ_CHI)


def cn_log_cut(cn: float, cnmax: float = EEQ_CN_MAX) -> float:
    """Плавное ограничение ЧК (mctc-lib ``log_cn_cut``).

    ``ln(1+e^cnmax) − ln(1+e^(cnmax−cn))``: асимптотически → cnmax при
    cn→∞, → cn при cn≪cnmax (не ``min(cn, cnmax)``).
    """
    return math.log1p(math.exp(cnmax)) - math.log1p(math.exp(cnmax - cn))


class EEQ2019:
    """EEQ-заряды eeq2019: ЧК (erf, cutoff 25, предел 8) + линейная система.

    Аргументы:
        zs: атомные номера (1..103 — дальше таблиц EEQ нет).
        pos_bohr: координаты (nat, 3) в bohr.
        charge: полный заряд системы (по умолчанию 0).
    """

    def __init__(
        self,
        zs: Sequence[int],
        pos_bohr: np.ndarray,
        charge: float = 0.0,
    ) -> None:
        """Инициализация: проверка входа, поэлементные параметры EEQ."""
        self.zs = [int(z) for z in zs]
        for z in self.zs:
            if not 1 <= z <= MAX_Z:
                msg = f"EEQ: атомный номер {z} вне области 1..{MAX_Z} (таблицы eeq2019)"
                raise ValueError(msg)
        self.pos = np.asarray(pos_bohr, dtype=float)
        if self.pos.ndim != 2 or self.pos.shape[0] != len(self.zs):
            msg = f"EEQ2019: размерность pos {self.pos.shape} не совпадает с nat={len(self.zs)}"
            raise ValueError(msg)
        self.nat = len(self.zs)
        self.charge = float(charge)

        self.chi = np.array([d4d.D4_EEQ_CHI[z - 1] for z in self.zs])
        self.eta = np.array([d4d.D4_EEQ_ETA[z - 1] for z in self.zs])
        self.kcnchi = np.array([d4d.D4_EEQ_KCNCHI[z - 1] for z in self.zs])
        self.rad = np.array([d4d.D4_EEQ_RAD[z - 1] for z in self.zs])
        # Радиус ЧК: (4/3)·covalent_rad_2009 (bohr) — те же таблицы, что у C6.
        self.rcov = np.array(
            [(4.0 / 3.0) * d4d.COVALENT_RAD_ANGSTROM[z - 1] * d4d.AATOA for z in self.zs]
        )

    def cn_raw(self) -> np.ndarray:
        """ЧК до log-ограничения (внутреннее: для производных)."""
        n = self.nat
        cn = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j:
                    continue
                r = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if r > EEQ_CN_CUTOFF:
                    continue
                rc = self.rcov[i] + self.rcov[j]
                s += 0.5 * (1.0 + math.erf(-EEQ_KCN * (r - rc) / rc))
            cn[i] = s
        return cn

    def cn(self) -> np.ndarray:
        """ЧК зарядовой модели: erf-счётчик (без EN-фактора) + предел 8.

        ``CN_i = ln-предельно( Σ_{j≠i, r≤25} ½(1+erf(−7.5(r−rc)/rc)) )``,
        ``rc = rcov_i + rcov_j`` (bohr).
        """
        return np.array([cn_log_cut(x) for x in self.cn_raw()])

    def dcn_dr(self) -> np.ndarray:
        """``dcn_dr[a, n, ic] = ∂CN_n/∂R_a^ic`` (CN с log-ограничением).

        Ограничение применяется после суммирования пар:
        ``∂CN_n = dlog(CN_raw_n)·∂CN_raw_n``, где
        ``dlog = e⁸/(e+e^CN)`` (mctc-lib ``dlog_cn_cut``).
        """
        n = self.nat
        cn_raw = self.cn_raw()
        dc = np.zeros((n, n, 3))
        for m in range(n):  # m — смещаемый атом
            for nm in range(n):  # nm — индекс ЧК
                dlog = math.exp(EEQ_CN_MAX) / (math.exp(EEQ_CN_MAX) + math.exp(cn_raw[nm]))
                rcov_nm = self.rcov[nm]
                for partner in range(n):
                    if partner == nm:
                        continue
                    rij = self.pos[nm] - self.pos[partner]
                    r = float(np.linalg.norm(rij))
                    if r > EEQ_CN_CUTOFF or r < 1e-6:
                        continue
                    rc = rcov_nm + self.rcov[partner]
                    x = EEQ_KCN * (r - rc) / rc
                    dcount = -EEQ_KCN * math.exp(-(x * x)) / (math.sqrt(math.pi) * rc)
                    # ∂CN_raw_nm/∂R_nm = +dcount·(R_nm−R_partner)/r,
                    # ∂CN_raw_nm/∂R_partner = −dcount·(R_nm−R_partner)/r
                    # (mctc ncoord_d: dcndr(:, iat, jat) += countd, rij = R_iat−R_jat)
                    vec = dcount * rij / r
                    if partner == m:
                        dc[m, nm] -= vec * dlog
                    elif nm == m:
                        dc[m, nm] += vec * dlog
        return dc

    def _coulomb_matrix(self) -> np.ndarray:
        """Матрица A (nat+1)×(nat+1) с ограничением Σq=Q (get_amat_0d)."""
        n = self.nat
        ndim = n + 1
        a = np.zeros((ndim, ndim))
        for i in range(n):
            a[i, i] = self.eta[i] + _SQRT2PI / self.rad[i]
        for i in range(n):
            for j in range(i + 1, n):
                r2 = float(np.sum((self.pos[i] - self.pos[j]) ** 2))
                gam = 1.0 / (self.rad[i] ** 2 + self.rad[j] ** 2)
                a[i, j] = a[j, i] = math.erf(math.sqrt(r2 * gam)) / math.sqrt(r2)
        a[n, :n] = 1.0
        a[:n, n] = 1.0
        a[n, n] = 0.0
        return a

    def _xvec(self, cn: np.ndarray) -> np.ndarray:
        """Правая часть x (get_xvec): x_i = −χ_i + kcnchi_i·CN_i/√(CN_i+reg)."""
        n = self.nat
        x = np.empty(n + 1)
        for i in range(n):
            x[i] = -self.chi[i] + self.kcnchi[i] * cn[i] / math.sqrt(cn[i] + _REG)
        x[n] = self.charge
        return x

    def charges(self) -> np.ndarray:
        """EEQ-заряды: решение ``A·[q; λ] = x`` (nat+1 уравнений)."""
        return np.linalg.solve(self._coulomb_matrix(), self._xvec(self.cn()))[: self.nat]

    def charges_and_lambda(self) -> tuple[np.ndarray, float]:
        """Заряды и множитель Лагранжа λ (нужен для CP-градиента)."""
        y = np.linalg.solve(self._coulomb_matrix(), self._xvec(self.cn()))
        return y[: self.nat], float(y[self.nat])

    def dqdr(self) -> np.ndarray:
        """``dqdr[a, j, ic] = ∂q_j/∂R_a^ic`` — CP-решение (multicharge solve).

        Связанные возмущения: ``A·y_a = dx_a − (∂A/R_a)·[q; λ]``,
        ``y_a = A⁻¹·(…)``; явный обратный A⁻¹ нужен (sytri в исходнике).
        Ограничительная строка x_{n+1}=Q не зависит от координат.
        """
        n = self.nat
        ndim = n + 1
        cn = self.cn()
        dcn = self.dcn_dr()
        a = self._coulomb_matrix()
        ainv = np.linalg.inv(a)
        q, lam = self.charges_and_lambda()
        qfull = np.append(q, lam)
        dxdr = np.zeros((n, ndim, 3))
        for nm in range(n):
            tmp = self.kcnchi[nm] / math.sqrt(cn[nm] + _REG)
            for ic in range(3):
                dxdr[:, nm, ic] = 0.5 * tmp * dcn[:, nm, ic]
        out = np.zeros((n, n, 3))
        for m in range(n):  # смещаемый атом
            dad = np.zeros((3, ndim, ndim))
            for partner in range(n):
                if partner == m:
                    continue
                rij = self.pos[partner] - self.pos[m]
                r2 = float(rij @ rij)
                r = math.sqrt(r2)
                gam = 1.0 / (self.rad[m] ** 2 + self.rad[partner] ** 2)
                u2 = gam * r2
                # dtmp = dA_ij/d(r²)·2 (eeq.f90 get_damat_0d); ∂A/R_m = −dtmp·rij
                dtmp = 2.0 * math.sqrt(gam) * math.exp(-u2) / (math.sqrt(math.pi) * r2) - math.erf(
                    math.sqrt(u2)
                ) / (r2 * r)
                for ic in range(3):
                    dad[ic, m, partner] = dad[ic, partner, m] = -dtmp * rij[ic]
            for ic in range(3):
                b = dxdr[m, :, ic] - dad[ic] @ qfull
                out[m, :, ic] = (ainv @ b)[:n]
        return out
