import numpy as np
from pyscf import dft

CF = (3 * np.pi**2) ** (2 / 3) / 10.0
D = 2.8

def pbe(n, sigma):
    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[n], [gx], [0.0], [0.0]])
    out = dft.libxc.eval_xc("GGA_C_PBE", raw)
    return out[0][0], out[1][0], out[1][1]

def pbc(n, sigma, tau):
    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[n], [gx], [0.0], [0.0], [tau]])
    out = dft.libxc.eval_xc("MGGA_C_TPSS", raw)
    return out[0][0], out[1][0], out[1][1], out[1][2]

n = 0.7
g, gr, gs = pbe(n, 0.0)
e, vr, vs, vt = pbc(n, 0.0, CF * n**2)
print(f"uniform: g={g:.12e}  pbc={e:.12e}  ratio={e/g:.12f}")

tau = 1.25 * CF * n**2
sig_iso = 8.0 * n * tau
e, vr, vs, vt = pbc(n, sig_iso, tau)
print(f"iso-orbital: pbc={e:.6e} (должно быть ~0)")

for zc in (0.1, 0.3, 0.6, 0.9, 1.5):
    tau = CF * n**2 * 1.0
    sig = zc * 8.0 * n * tau
    g, gr, gs = pbe(n, sig)
    e, vr, vs, vt = pbc(n, sig, tau)
    u = min(zc, 1.0)
    pred = g * (1 - u**2) * (1 + D * g * (1 - u**2) * u**3)
    print(f"Z={zc:4.2f}: g={g:.10e} pbc={e:.10e} pred(U=Z)={pred:.10e} diff={abs(e - pred):.3e}")
