import numpy as np
from pyscf import dft

GAMMA = (1 - np.log(2)) / np.pi**2
BETA = 0.06672455060314922
PWA = (0.0310907, 0.01554535, 0.0168869)
PWA1 = (0.21370, 0.20548, 0.11125)
PB1 = (7.5957, 14.1189, 10.357); PB2 = (3.5876, 6.1977, 3.6231)
PB3 = (1.6382, 3.3662, 0.88026); PB4 = (0.49294, 0.62517, 0.49671)
FZ20 = 1.709920934161365617563962776245
D_PBC = 2.8
C0 = 0.53
CF = (3 * np.pi**2) ** (2 / 3) / 10.0

def _g_k(k, rs):
    p = PB1[k]*np.sqrt(rs) + PB2[k]*rs + PB3[k]*rs**1.5 + PB4[k]*rs**2
    return -2*PWA[k]*(1+PWA1[k]*rs)*np.log1p(1/(2*PWA[k]*p))

def f_pw_zeta(rs, z):
    fz = ((1+z)**(4/3) + (1-z)**(4/3) - 2)/(2**(4/3) - 2)
    g0, g1, g2 = (_g_k(k, rs) for k in range(3))
    return g0 + z**4*fz*(g1 - g0 + g2/FZ20) - fz*g2/FZ20

def f_pbe_zeta(rs, z, xt2):
    mphi = ((1+z)**(2/3) + (1-z)**(2/3))/2
    f_lda = f_pw_zeta(rs, z)
    E = np.expm1(-f_lda/(GAMMA*mphi**3))
    A = BETA/(GAMMA*E)
    t2 = xt2/(16*2**(2/3)*mphi**2*rs)
    f1 = t2 + A*t2**2
    g = 1/(1 + A*f1)
    return f_lda + GAMMA*mphi**3*np.log1p((1-g)*E)

def pbc_c(rho, sigma, tau):
    rs = (3/(4*np.pi*rho))**(1/3)
    xt2 = sigma/rho**(8/3)
    t_total = 2**(2/3)*tau/rho**(5/3)
    u = min(xt2/(8*t_total), 1.0)
    g_tot = f_pbe_zeta(rs, 0.0, xt2)
    g_p = f_pbe_zeta(rs*2**(1/3), 1.0, xt2)
    g_m = f_pbe_zeta(rs*2**(1/3), -1.0, xt2)
    f0 = g_tot*(1 + C0*u**2) - (1 + C0)*u**2*0.5*(g_p + g_m)
    return f0*(1 + D_PBC*f0*u**3)

def ref(rho, sigma, tau):
    gx = np.sqrt(sigma) if sigma > 0 else 0.0
    raw = np.array([[rho], [gx], [0.0], [0.0], [tau]])
    return dft.libxc.eval_xc("MGGA_C_TPSS", raw)[0][0]

# 1) граничные случаи
rho, tau = 0.7, CF*0.7**2
e = ref(rho, 0.0, tau)
g0 = f_pbe_zeta((3/(4*np.pi*rho))**(1/3), 0.0, 0.0)
print("sigma=0:  ref", e, " f_pbe(z=0,xt2=0)", g0, "diff", abs(e - g0))

rho, sigma, tau = 0.7, 0.3, 0.6
e_ref = ref(rho, sigma, tau)
e_mine = pbc_c(rho, sigma, tau)
print("test pt: ref", e_ref, " mine", e_mine, "diff", abs(e_ref - e_mine))

# 2) случайная сетка
rng = np.random.default_rng(123)
worst = 0.0
for i in range(2000):
    r = 10**rng.uniform(-2, 0.8)
    t = CF*r**2*10**rng.uniform(-1.2, 1.2)
    zc = 10**rng.uniform(-3, 1.1)   # = xt2/(8 t_total) = sigma/(8*2^(2/3)*r*tau)
    s = zc*8*2**(2/3)*r*t
    if s > 0 and t > 1e-9:
        d = abs(ref(r, s, t) - pbc_c(r, s, t))
        if d > worst:
            worst = d
            wpt = (r, s, t)
print("worst abs diff over grid:", worst, "at", wpt if worst else None)
