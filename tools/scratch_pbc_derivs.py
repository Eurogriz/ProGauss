"""PBC (MGGA_C_TPSS, z=0) derivative convention probe.

Determines how the bundled libxc returns vrho/vsigma/vtau in the vW-clamped
region (sigma > 8*rho*tau, i.e. U > 1):

- convention A: driver rewrites sigma -> se = min(sigma, 8*rho*tau); the kernel
  returns unclamped partials of e(rho, se, tau).
- convention C: U = min(U,1) clamped INSIDE the kernel; t2 = sigma/rho^{8/3}
  stays unclamped; partials w.r.t. sigma flow through t2 only, dU/dsigma = 0
  when clamped.

Energy for both equals the verified pbc_c (scratch_pbc_final.py).
"""
from __future__ import annotations

import numpy as np
import sympy as sp

GAMMA = (1.0 - np.log(2.0)) / np.pi**2
BETA = 0.06672455060314922

rho, s, tau = sp.symbols("rho s tau", positive=True)
gamma, beta = sp.symbols("gamma beta", positive=True)

FZ20 = sp.Float("1.709920934161365617563962776245")
A_ = (sp.Float("0.0310907"), sp.Float("0.01554535"), sp.Float("0.0168869"))
AL = (sp.Float("0.21370"), sp.Float("0.20548"), sp.Float("0.11125"))
B1 = (sp.Float("7.5957"), sp.Float("14.1189"), sp.Float("10.357"))
B2 = (sp.Float("3.5876"), sp.Float("6.1977"), sp.Float("3.6231"))
B3 = (sp.Float("1.6382"), sp.Float("3.3662"), sp.Float("0.88026"))
B4 = (sp.Float("0.49294"), sp.Float("0.62517"), sp.Float("0.49671"))


def f_pw_sym(rs, z):
    gs = [
        -2 * a * (1 + al * rs) * sp.log(1 + 1 / (2 * a * (b1 * sp.sqrt(rs) + b2 * rs + b3 * rs**sp.Rational(3, 2) + b4 * rs**2)))
        for a, al, b1, b2, b3, b4 in zip(A_, AL, B1, B2, B3, B4)
    ]
    fz = ((1 + z) ** sp.Rational(4, 3) + (1 - z) ** sp.Rational(4, 3) - 2) / (2 * 2**sp.Rational(1, 3) - 2)
    return gs[0] + z**4 * fz * (gs[1] - gs[0] + gs[2] / FZ20) - fz * gs[2] / FZ20


def f_pbe_c_sym(rs, t2, z):
    m = ((1 + z) ** sp.Rational(2, 3) + (1 - z) ** sp.Rational(2, 3)) / 2
    flda = f_pw_sym(rs, z)
    E = sp.exp(-flda / (gamma * m**3)) - 1
    A = beta / (gamma * E)
    tt2 = t2 / (16 * 2**sp.Rational(2, 3) * m**2 * rs)
    f1 = tt2 + A * tt2**2
    gsat = 1 / (1 + A * f1)
    return flda + gamma * m**3 * sp.log(1 + (1 - gsat) * E)


# --- convention A: everything evaluated at se (unclamped formula) ---
rs = (sp.Rational(3, 1) / (4 * sp.pi * rho))**sp.Rational(1, 3)
U = s / (8 * rho * tau)
t2 = s * rho**sp.Rational(-8, 3)
rs_c = rs * 2**sp.Rational(1, 3)
xs02 = 2**sp.Rational(2, 3) * t2
g = f_pbe_c_sym(rs, t2, 0)
cp = f_pbe_c_sym(rs_c, xs02, 1)
cm = f_pbe_c_sym(rs_c, xs02, -1)

PAIRS_A = {"cc": (cp, cm), "cg": (cp, g), "gc": (g, cm), "gg": (g, g)}


def EA(bp, bm):
    f0 = (1 + sp.Float("0.53") * U**2) * g - sp.Float("1.53") * U**2 * (bp + bm) / 2
    return f0 * (1 + sp.Float("2.8") * f0 * U**3)


# --- convention C: U fixed at 1, t2 unclamped (clamped region only) ---
t2r = s * rho**sp.Rational(-8, 3)  # same form, but NOT replaced by se
gC = f_pbe_c_sym(rs, t2r, 0)
cpC = f_pbe_c_sym(rs * 2**sp.Rational(1, 3), 2**sp.Rational(2, 3) * t2r, 1)
cmC = f_pbe_c_sym(rs * 2**sp.Rational(1, 3), 2**sp.Rational(2, 3) * t2r, -1)
PAIRS_C = {"cc": (cpC, cmC), "cg": (cpC, gC), "gc": (gC, cmC), "gg": (gC, gC)}


def EC(bp, bm):
    f0 = (1 + sp.Float("0.53")) * gC - sp.Float("1.53") * (bp + bm) / 2
    return f0 * (1 + sp.Float("2.8") * f0)


LAM = {}
for conv, gfun, pairs, Efun in (("A", g, PAIRS_A, EA), ("C", gC, PAIRS_C, EC)):
    LAM[conv] = {}
    for name, (bp, bm) in pairs.items():
        ee = Efun(bp, bm)
        LAM[conv][name] = (
            sp.lambdify((rho, s, tau, gamma, beta), ee, "numpy"),
            sp.lambdify((rho, s, tau, gamma, beta), sp.diff(ee, rho), "numpy"),
            sp.lambdify((rho, s, tau, gamma, beta), sp.diff(ee, s), "numpy"),
            sp.lambdify((rho, s, tau, gamma, beta), sp.diff(ee, tau), "numpy"),
        )

LgA = sp.lambdify((rho, s, tau, gamma, beta), g, "numpy")
LcpA = sp.lambdify((rho, s, tau, gamma, beta), cp, "numpy")
LcmA = sp.lambdify((rho, s, tau, gamma, beta), cm, "numpy")
LgC = sp.lambdify((rho, s, tau, gamma, beta), gC, "numpy")
LcpC = sp.lambdify((rho, s, tau, gamma, beta), cpC, "numpy")
LcmC = sp.lambdify((rho, s, tau, gamma, beta), cmC, "numpy")


def _branch(Lg, Lcp, Lcm, r, ss, t):
    gv = float(Lg(r, ss, t, GAMMA, BETA))
    cpv = float(Lcp(r, ss, t, GAMMA, BETA))
    cmv = float(Lcm(r, ss, t, GAMMA, BETA))
    return ("c" if cpv >= gv else "g") + ("c" if cmv >= gv else "g")


def evalA(r, sigma, t):
    se = min(sigma, 8 * r * t)
    name = _branch(LgA, LcpA, LcmA, r, se, t)
    le, dr, ds, dt = LAM["A"][name]
    e = float(le(r, se, t, GAMMA, BETA))
    return e + r * float(dr(r, se, t, GAMMA, BETA)), r * float(ds(r, se, t, GAMMA, BETA)), r * float(dt(r, se, t, GAMMA, BETA))


def evalC(r, sigma, t):
    name = _branch(LgC, LcpC, LcmC, r, sigma, t)
    le, dr, ds, dt = LAM["C"][name]
    e = float(le(r, sigma, t, GAMMA, BETA))
    return e + r * float(dr(r, sigma, t, GAMMA, BETA)), r * float(ds(r, sigma, t, GAMMA, BETA)), r * float(dt(r, sigma, t, GAMMA, BETA))


def oracle(r, sigma, t):
    from pyscf import dft

    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[r], [gx], [0.0], [0.0], [t]])
    ex, vxc, _, _ = dft.libxc.eval_xc("MGGA_C_TPSS", raw, deriv=1)
    return ex[0], vxc[0][0], vxc[1][0], vxc[3][0]


if __name__ == "__main__":
    pts = [
        (1.0, 1.6, 0.1),
        (1.0, 8.0, 0.1),
        (0.6921949562710633, 1.235228096797341, 0.16728526816170367),
        (3.5320293902192823, 1203.9413179081523, 2.8962070879974187),
        (0.5, 0.8, 0.1),
    ]
    for (r, sig, t) in pts:
        oe, ov, ovS, ovT = oracle(r, sig, t)
        va, sa, ta = evalA(r, sig, t)
        vc, sc, tc = evalC(r, sig, t)
        print(f"(r={r}, sig={sig}, t={t}) clamped={sig > 8 * r * t}")
        print(f"  oracle: vr={ov:+.6e} vs={ovS:+.6e} vt={ovT:+.6e}")
        print(f"  A:      vr={va:+.6e} vs={sa:+.6e} vt={ta:+.6e}  d=({va-ov:+.2e},{sa-ovS:+.2e},{ta-ovT:+.2e})")
        print(f"  C:      vr={vc:+.6e} vs={sc:+.6e} vt={tc:+.6e}  d=({vc-ov:+.2e},{sc-ovS:+.2e},{tc-ovT:+.2e})")
