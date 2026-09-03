"""Вывод аналитических производных meta-GGA TPSS-x и PBC-c (z=0, unpol,
НЕклампованное ядро) через SymPy и сверка с оракулом (pyscf 2.14.0, libxc
7.0.0 build) на физической области sigma < 8*rho*tau.

Решение о конвенции (см. SESSION_MEMORY.md):
- для физических плотностей (многэлектронные детерминанты: sigma < 8*rho*tau)
  оракульский libxc НЕ клампит sigma (драйверный clamp неактивен), поэтому
  неклампованное ядро = оракульские значения ТОЧНО (до машиной точности);
- в нефизической области sigma > 8*rho*tau oракульский 7.0.0-build клампит
  sigma, и его производные — известный дефект libxc (MR !800): мы его не
  воспроизводим и в docstring функционала это задокументировано.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

# -------------------------------------------------------------------------- #
# Общие константы
# -------------------------------------------------------------------------- #
GAMMA = (1.0 - np.log(2.0)) / np.pi**2
BETA = 0.06672455060314922

# TPSS-x параметры (сверены с оракулом ранее, 500 точек, 8.9e-16):
TPSS_B = 0.40
TPSS_C = 1.59096
TPSS_E = 1.537
TPSS_KAPPA = 0.8040
TPSS_MU = 0.21951
TPSS_BLOC_A = 2.0
TPSS_BLOC_B = 0.0

# PW92-модифицированные константы PBE-c/PBC (из scratch_pbc_final.py,
# сверены с оракулом: worst 1.23e-12 на энергии).
PW92_A = (0.0310907, 0.01554535, 0.0168869)
PW92_AL1 = (0.21370, 0.20548, 0.11125)
PW92_B1 = (7.5957, 14.1189, 10.357)
PW92_B2 = (3.5876, 6.1977, 3.6231)
PW92_B3 = (1.6382, 3.3662, 0.88026)
PW92_B4 = (0.49294, 0.62517, 0.49671)
PW92_FZ20 = 1.709920934161365617563962776245


# -------------------------------------------------------------------------- #
# TPSS-x: энергия на единицу объёма (транскрипция сгенерированного C-кода
# libxc 7.0.0, mgga_x_tpss.c, unpol, сырой sigma).
# -------------------------------------------------------------------------- #
def build_tpssx() -> dict[str, sp.Expr]:
    """Построчная транскрипция func_exc_unpol (mgga_x_tpss.c, libxc 7.0.0).

    Только unpol (z=0): спин-фактор t18 = 1, ветка dens_threshold не активна
    (по умолчанию 0.01; здесь rho >= 1e-3 в верификации, и даже при срабатывании
    формула одна).
    """
    R = sp.Rational
    rho, sigma, tau = sp.symbols("rho sigma tau", positive=True)
    b, c, e, kappa, mu = (TPSS_B, TPSS_C, TPSS_E, TPSS_KAPPA, TPSS_MU)

    t4 = 3 ** sp.Rational(1, 3)            # M_CBRT3
    t5 = sp.pi ** sp.Rational(1, 3)        # M_CBRTPI
    t7 = t4 / t5                           # (3/pi)^(1/3)
    t19 = rho ** sp.Rational(1, 3)
    t20 = t19                              # t18 (unpol) * t19
    t21 = 1.0 / rho
    t23 = 1.0 / tau
    t25 = sigma * t21 * t23 / 8.0          # U = sigma/(8 rho tau)
    t30 = TPSS_BLOC_A + TPSS_BLOC_B * sigma * t21 * t23 / 8.0
    t31 = t25**t30
    t32 = c * t31
    t33 = sigma * sigma
    t34 = rho * rho
    t35 = 1.0 / t34
    t36 = t33 * t35
    t37 = tau * tau
    t38 = 1.0 / t37
    t39 = t36 * t38
    t41 = 1.0 + t39 / 64.0
    t42 = t41 * t41
    t43 = 1.0 / t42
    t46 = 6 ** sp.Rational(1, 3)
    t47 = (R(10, 81) + t32 * t43) * t46
    t48 = sp.pi * sp.pi
    t49 = t48 ** sp.Rational(1, 3)
    t50 = t49 * t49
    t51 = 1.0 / t50
    t52 = t47 * t51
    t53 = 2 ** sp.Rational(1, 3)
    t54 = t53 * t53
    t55 = sigma * t54
    t56 = t19 * t19
    t58 = 1.0 / t56 / t34
    t59 = t55 * t58                        # S
    t62 = tau * t54
    t64 = 1.0 / t56 / rho
    t67 = t62 * t64 - t59 / 8.0
    t71 = R(5, 9) * t67 * t46 * t51 - 1.0  # alpha - 1, alpha = 1 + t67/K_FACTOR_C
    # ВАЖНО (2026-09-02): сгенерированный C в архиве gitlab 7.0.0 устарел
    # (0.5e1*t72*t74 + 0.9e1 = 9(1+b(alpha-1)^2)), а .so-оракул использует
    # формулу из maple-источника sqrt(1 + b*alpha*(alpha-1)). Проверка на
    # 40 точках: оракульная форма 4.3e-16, «архивная» — 4.6e-3.
    t77 = 9.0 * (1.0 + b * (1.0 + t71) * t71)
    t73 = t46 * t51
    t78 = sp.sqrt(t77)
    t79 = 1.0 / t78
    # 0.27e2/0.2e2 = 27/20 = 1.35 (не 27/2!).
    t84 = R(27, 20) * t71 * t79 + t73 * t59 / 36.0
    t85 = t84 * t84
    t88 = t46 * t46
    t90 = 1.0 / t49 / t48
    t91 = t88 * t90
    t92 = t33 * t53
    t93 = t34 * t34
    t94 = t93 * rho
    t96 = 1.0 / t19 / t94
    t97 = t92 * t96
    t100 = 100.0 * t91 * t97 + 162.0 * t39
    t101 = sp.sqrt(t100)
    t105 = 1.0 / kappa * t88
    t106 = t105 * t90
    t109 = sp.sqrt(e)
    t110 = t109 * t33
    t111 = t35 * t38
    t114 = e * mu
    t115 = t48 * t48
    t116 = 1.0 / t115
    t117 = t33 * sigma
    t118 = t116 * t117
    t119 = t93 * t93
    t120 = 1.0 / t119
    t124 = (
        t52 * t59 / 24.0
        + R(146, 2025) * t85
        - R(73, 97200) * t84 * t101
        + R(25, 472392) * t106 * t97
        + t110 * t111 / 720.0
        + t114 * t118 * t120 / 576.0
    )
    t125 = t109 * t46
    t129 = 1.0 + t125 * t51 * t59 / 24.0
    t130 = t129 * t129
    t131 = 1.0 / t130
    t133 = t124 * t131 + kappa
    t138 = 1.0 + kappa * (1.0 - kappa / t133)
    # В C-коде tzk0 = 2*t142: ядро считает один спин-канал, а вывод для unpol
    # удваивается (два канала z=±0). Полная энергия на частицу:
    # eps_pp = -(3/4)(3/pi)^(1/3) rho^(1/3) F.
    t142 = -R(3, 4) * t7 * t20 * t138
    return {"E_V": rho * t142, "eps_pp": t142, "symbols": (rho, sigma, tau)}


# -------------------------------------------------------------------------- #
# PBC: энергия на единицу объёма (unpol, z=0, сырой u = sigma/(8 rho tau),
# t2 = sigma/rho^(8/3) без vW-clamp).
# -------------------------------------------------------------------------- #
def _f_pw_mod_sympy(rs: sp.Expr, z: sp.Expr) -> sp.Expr:
    gz = []
    for a, al, b1, b2, b3, b4 in zip(PW92_A, PW92_AL1, PW92_B1, PW92_B2, PW92_B3, PW92_B4):
        poly = b1 * sp.sqrt(rs) + b2 * rs + b3 * rs**sp.Rational(3, 2) + b4 * rs**2
        gz.append(-2 * a * (1 + al * rs) * sp.log(1 + 1 / (2 * a * poly)))
    fz = ((1 + z) ** sp.Rational(4, 3) + (1 - z) ** sp.Rational(4, 3) - 2) / (
        2 * 2 ** sp.Rational(1, 3) - 2
    )
    return gz[0] + z**4 * fz * (gz[1] - gz[0] + gz[2] / PW92_FZ20) - fz * gz[2] / PW92_FZ20


def _f_pbe_c_sympy(rs: sp.Expr, z: sp.Expr, t2: sp.Expr) -> sp.Expr:
    mphi = ((1 + z) ** sp.Rational(2, 3) + (1 - z) ** sp.Rational(2, 3)) / 2
    f_lda = _f_pw_mod_sympy(rs, z)
    E = sp.exp(-f_lda / (GAMMA * mphi**3)) - 1  # x >= 0.7: потери expm1 не критичны
    A = BETA / (GAMMA * E)
    tt2 = t2 / (16 * 2 ** sp.Rational(2, 3) * mphi**2 * rs)
    f1 = tt2 + A * tt2**2
    g_sat = 1 / (1 + A * f1)
    return f_lda + GAMMA * mphi**3 * sp.log(1 + (1 - g_sat) * E)


def build_pbc() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Возвращает (E_V, g, br-ветви) — сырые выражения от (rho, sigma, tau)."""
    rho, sigma, tau = sp.symbols("rho sigma tau", positive=True)
    rs = (3 / (4 * sp.pi * rho)) ** sp.Rational(1, 3)
    t2 = sigma * rho ** sp.Rational(-8, 3)
    u = sigma / (8 * rho * tau)
    g = _f_pbe_c_sympy(rs, 0, t2)
    chan_p = _f_pbe_c_sympy(rs * 2 ** sp.Rational(1, 3), 1, 2 ** sp.Rational(2, 3) * t2)
    chan_m = _f_pbe_c_sympy(rs * 2 ** sp.Rational(1, 3), -1, 2 ** sp.Rational(2, 3) * t2)
    return rho, sigma, tau, rs, t2, u, g, chan_p, chan_m


def pbc_energy_parts(rho, sigma, tau, g, chan_p, chan_m):
    """f0 = (1+0.53u^2)g - 1.53u^2*br, e_V = rho*f0*(1+2.8 f0 u^3).

    br = 0.5*max(chan_p, g) + 0.5*max(chan_m, g): ветки max выбираются в
    момент вычисления (в производной — та же ветка; в точке пересечения
    веток производная не определена — мера ноль).
    """
    u = sigma / (8 * rho * tau)
    br_p = sp.Max(chan_p, g)
    br_m = sp.Max(chan_m, g)
    f0 = (1 + 0.53 * u**2) * g - 1.53 * u**2 * (br_p + br_m) / 2
    return f0, u


# -------------------------------------------------------------------------- #
# Численная часть
# -------------------------------------------------------------------------- #
def oracle(xc_name: str, rho: float, sigma: float, tau: float) -> np.ndarray:
    from pyscf import dft

    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[rho], [gx], [0.0], [0.0], [tau]])
    res = dft.libxc.eval_xc(xc_name, raw, deriv=1)
    e = np.asarray(res[0]).ravel()[0]
    drho = np.asarray(res[1][0]).ravel()[0]
    dsigma = np.asarray(res[1][1]).ravel()[0]
    dtau = np.asarray(res[1][3]).ravel()[0]
    return np.array([e, drho, dsigma, dtau])


def physical_points(rng: np.random.Generator, n: int) -> list[tuple[float, float, float]]:
    pts = []
    for _ in range(n):
        r = 10.0 ** rng.uniform(-2.5, 1.0)
        t = 3.0 * r**2 * 10.0 ** rng.uniform(-1.5, 1.5)
        # sigma < 8*rho*tau с гарантированным запасом (физический регион).
        s = 8.0 * r * t * 10.0 ** rng.uniform(-2.5, -0.05)
        pts.append((r, s, t))
    return pts


def verify(tpssx: dict[str, sp.Expr], pbc_parts) -> None:
    # TPSS-x: выражение уже от (rho, sigma, tau). ВАЖНО: дифференцировать по
    # ТЕМ ЖЕ объектам символов (sp.diff по строке — другой символ -> 0!).
    rho_s, sigma_s, tau_s = tpssx["symbols"]
    tpssx_ev2 = tpssx["E_V"]
    d_rho_x = sp.diff(tpssx_ev2, rho_s)
    d_s_x = sp.diff(tpssx_ev2, sigma_s)
    d_t_x = sp.diff(tpssx_ev2, tau_s)
    f_x = sp.lambdify((rho_s, sigma_s, tau_s), [tpssx_ev2, d_rho_x, d_s_x, d_t_x], modules="numpy")

    # PBC.
    (rho, sigma, tau, rs, t2, u, g, chan_p, chan_m) = pbc_parts
    f0, _u = pbc_energy_parts(rho, sigma, tau, g, chan_p, chan_m)
    e_v_pbc = rho * f0 * (1 + 2.8 * f0 * u**3)
    d_rho_c = sp.diff(e_v_pbc, rho)
    d_s_c = sp.diff(e_v_pbc, sigma)
    d_t_c = sp.diff(e_v_pbc, tau)
    f_c = sp.lambdify(("rho", "sigma", "tau"), [e_v_pbc, d_rho_c, d_s_c, d_t_c], modules="numpy")

    rng = np.random.default_rng(77)
    pts = physical_points(rng, 250) + physical_points(np.random.default_rng(78), 250)
    worst = {"x": (0, None), "c": (0, None)}
    for r, s, t in pts:
        ref = oracle("MGGA_X_TPSS", r, s, t)
        mine = np.asarray(f_x(r, s, t), dtype=float)
        mine[0] = mine[0] / r  # E_V -> eps_pp
        err = np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-30))
        if err > worst["x"][0]:
            worst["x"] = (err, (r, s, t))
        ref = oracle("MGGA_C_TPSS", r, s, t)
        mine = np.asarray(f_c(r, s, t), dtype=float)
        mine[0] = mine[0] / r  # E_V -> eps_pp
        err = np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-30))
        if err > worst["c"][0]:
            worst["c"] = (err, (r, s, t))
    print(f"TPSS-x worst rel = {worst['x'][0]:.3e} at {worst['x'][1]}")
    print(f"PBC-c  worst rel = {worst['c'][0]:.3e} at {worst['c'][1]}")


if __name__ == "__main__":
    print("building TPSS-x...")
    tpssx = build_tpssx()
    print("building PBC...")
    pbc_parts = build_pbc()
    verify(tpssx, pbc_parts)
