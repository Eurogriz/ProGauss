import numpy as np
from scipy.optimize import brentq
from pyscf import dft

CF = (3 * np.pi**2) ** (2 / 3) / 10.0
D = 2.8

def pbe(n, sigma):
    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[n], [gx], [0.0], [0.0]])
    return dft.libxc.eval_xc("GGA_C_PBE", raw)[0][0]

def pbc(n, sigma, tau):
    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[n], [gx], [0.0], [0.0], [tau]])
    return dft.libxc.eval_xc("MGGA_C_TPSS", raw)[0][0]

def solve_u(e, g):
    # e = g(1-u^2)(1 + D g (1-u^2) u^3), u in [0,1]
    f = lambda u: g * (1 - u**2) * (1 + D * g * (1 - u**2) * u**3) - e
    a, b = 0.0, 1.0
    fa, fb = f(a), f(b)
    if fa == 0: return 0.0
    if fa * fb > 0: return np.nan
    return brentq(f, a, b, xtol=1e-15)

rng = np.random.default_rng(7)
rows = []
for i in range(40):
    n = 10 ** rng.uniform(-2, 0.6)
    tau = CF * n**2 * 10 ** rng.uniform(-1, 1)
    z_target = 10 ** rng.uniform(-2, 0.9)   # = x^2/(8 t_libxc) = sigma/(64 n tau)
    sigma = z_target * 64.0 * n * tau
    g = pbe(n, sigma)
    e = pbc(n, sigma, tau)
    u = solve_u(e, g)
    if not np.isfinite(u):
        continue
    rows.append((z_target, u))

z = np.array([r[0] for r in rows])
u = np.array([r[1] for r in rows])
print("z_libxc, u_ref, ratio u/z:")
for zi, ui in sorted(zip(z, u))[:8]:
    print(f"  z={zi:.6e}  u={ui:.6e}  u/z={ui/zi:.6f}")
rat = u / z
print(f"median ratio={np.median(rat):.6f}  min={rat.min():.6f} max={rat.max():.6f}")
