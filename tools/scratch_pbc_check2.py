import numpy as np
from pyscf import dft

CF = (3 * np.pi**2) ** (2 / 3) / 10.0
D = 2.8

def pbe(n, sigma):
    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[n], [gx], [0.0], [0.0]])
    out = dft.libxc.eval_xc("GGA_C_PBE", raw)
    return out[0][0]

def pbc(n, sigma, tau):
    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[n], [gx], [0.0], [0.0], [tau]])
    out = dft.libxc.eval_xc("MGGA_C_TPSS", raw)
    return out[0][0]

def pred(n, sigma, tau, factor):
    g = pbe(n, sigma)
    z = sigma / (64.0 * n * tau)  # x^2/(8t_libxc)
    u = min(z * factor, 1.0)
    f0 = g * (1 - u**2)
    return f0 * (1 + D * f0 * u**3)

rng = np.random.default_rng(42)
worst = {f: 0.0 for f in (1.0, 1/2**(2/3), 0.5, 2.0, 1/4, 2**(2/3))}
for i in range(300):
    n = 10 ** rng.uniform(-2, 0.6)
    tau = CF * n**2 * 10 ** rng.uniform(-1, 1)
    sigma = 64.0 * n * tau * min(10 ** rng.uniform(-2, 0.9), 1.5)
    e = pbc(n, sigma, tau)
    for f in worst:
        worst[f] = max(worst[f], abs(e - pred(n, sigma, tau, f)))
for f, w in sorted(worst.items(), key=lambda kv: kv[1]):
    print(f"factor={f:.6f}  worst_abs_diff={w:.3e}")
