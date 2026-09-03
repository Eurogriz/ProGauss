"""Финальная верификация PBC (MGGA_C_TPSS, z=0, unpol) против libxc 7.0.0 (pyscf).

Формула (полностью верифицирована численно, см. вывод):

    rs     = (3/(4*pi*rho))**(1/3)
    xt2    = sigma / rho**(8/3)              # reduced gradient^2 (total)
    t_kin  = tau / rho**(5/3)                # reduced kinetic energy
    U      = min(xt2/(8*t_kin), 1)           # clamp: tau < tau_W (vW) => U=1
    t2_eff = 8*t_kin*U  = min(xt2, 8*t_kin)  # H-term argument (vW-clamped)

    g      = f_pbe_c(rs, 0, t2_eff)
    chan_p = f_pbe_c(rs*2**(1/3),  +1, 2**(2/3)*t2_eff)
    chan_m = f_pbe_c(rs*2**(1/3),  -1, 2**(2/3)*t2_eff)
    br     = 0.5*max(chan_p, g) + 0.5*max(chan_m, g)
    f0     = (1 + 0.53*U^2)*g - (1 + 0.53)*U^2*br
    e_pp   = f0 * (1 + 2.8*f0*U^3)

f_pbe_c(rs, z, t2) = f_pw_mod(rs, z) + gamma*mphi(z)^3*log(1 + E*(1-g_sat)),
    E        = expm1(-f_pw/(gamma*mphi^3))
    A        = beta/(gamma*E)
    tt2      = t2/(16*2**(2/3)*mphi^2*rs)
    f1       = tt2 + A*tt2**2
    g_sat    = 1/(1 + A*f1)
    mphi(z)  = ((1+z)**(2/3) + (1-z)**(2/3))/2

f_pw_mod — PW92 correlation с МОДИФИЦИРОВАННЫМИ константами (совпадает с
LDA-ядром GGA_C_PBE / PBC, НЕ с LDA_C_PW):
    a    = (0.0310907, 0.01554535, 0.0168869)
    fz20 = 1.709920934161365617563962776245
    alpha1 = (0.21370, 0.20548, 0.11125)
    beta1 = (7.5957, 14.1189, 10.357),  beta2 = (3.5876, 6.1977, 3.6231)
    beta3 = (1.6382, 3.3662, 0.88026),  beta4 = (0.49294, 0.62517, 0.49671)
    f_pw = g1 + z^4*fz(z)*(g2 - g1 + g3/fz20) - fz(z)*g3/fz20
    g_k  = -2*a_k*(1+alpha1_k*rs)*log(1 + 1/(2*a_k*(beta1_k*sqrt(rs)
              + beta2_k*rs + beta3_k*rs**1.5 + beta4_k*rs**2)))
    fz(z) = ((1+z)**(4/3) + (1-z)**(4/3) - 2)/(2*2**(1/3) - 2)

gamma = (1 - ln 2)/pi^2,  beta = 0.06672455060314922.

Источники формулы:
  - структура f0/d-члена: libxc maple (tpss_c.mpl: tpss_f = tpss_f0*(1+d*tpss_f0*aux^3),
    d=2.8, C0_c[0]=0.53), сгенерированный код 5.1.1/7.0.0 (mgga_c_tpss.c:
    каналы = 2^(2/3)*xt2, C0=const при z=0).
  - clamp vW (t2_eff = min(xt2, 8*t_kin), U = min(xt2/(8 t_kin),1)):
    ВЫЯВЛЕН ЧИСЛЕННО (libxc 7.0.0: e(rho, s, tau) = e(rho, min(s, 8*tau*rho^(1/3)), tau)
    ТОЧНО при s > 8*tau*rho^(1/3); в 5.1.1 кода такого clampa не видно — 7.0.0
    является действующим оракулом (pyscf 2.14 bundling libxc 7.0.0)).
  - H-терм (f1 = tt2 + A*tt2^2, g_sat, ln(1+E(1-g_sat))): проверен тождественно
    GGA_C_PBE(7.0.0) на 500 точках (worst 9.7e-17, sigma до 1e3).
  - low-density screening-ветка (rho/2 <= dens_threshold) в 7.0.0 НЕ активна
    (default dens_threshold): non-screening формула верна вплоть до rho = 0.005.
"""
import numpy as np

GAMMA = (1.0 - np.log(2.0)) / np.pi**2
BETA = 0.06672455060314922

_PW92_MOD_A = np.array([0.0310907, 0.01554535, 0.0168869])
_PW92_MOD_AL1 = np.array([0.21370, 0.20548, 0.11125])
_PW92_MOD_B1 = np.array([7.5957, 14.1189, 10.357])
_PW92_MOD_B2 = np.array([3.5876, 6.1977, 3.6231])
_PW92_MOD_B3 = np.array([1.6382, 3.3662, 0.88026])
_PW92_MOD_B4 = np.array([0.49294, 0.62517, 0.49671])
_PW92_FZ20 = 1.709920934161365617563962776245


def f_pw_mod(rs, z):
    g = np.array([
        -2.0*a*(1.0 + al*rs)
        * np.log1p(1.0/(2.0*a*(b1*np.sqrt(rs) + b2*rs + b3*rs**1.5 + b4*rs**2)))
        for a, al, b1, b2, b3, b4 in zip(
            _PW92_MOD_A, _PW92_MOD_AL1, _PW92_MOD_B1, _PW92_MOD_B2, _PW92_MOD_B3, _PW92_MOD_B4
        )
    ])
    fz = (
        ((1.0 + z)**(4.0/3.0) + (1.0 - z)**(4.0/3.0) - 2.0) / (2.0*2.0**(1.0/3.0) - 2.0)
        if abs(z) < 1.0
        else (2.0**(4.0/3.0) - 2.0) / (2.0*2.0**(1.0/3.0) - 2.0)
    )
    return g[0] + z**4*fz*(g[1] - g[0] + g[2]/_PW92_FZ20) - fz*g[2]/_PW92_FZ20


def f_pbe_c(rs, z, t2):
    """PBE-c (PW92-modified + PBE H-терм) при z = +-1/0; t2 — reduced gradient^2."""
    mphi = ((1.0 + z)**(2.0/3.0) + (1.0 - z)**(2.0/3.0)) / 2.0
    f_lda = f_pw_mod(rs, z)
    E = np.expm1(-f_lda/(GAMMA*mphi**3))
    A = BETA/(GAMMA*E)
    tt2 = t2/(16.0*2.0**(2.0/3.0)*mphi**2*rs)
    f1 = tt2 + A*tt2**2
    g_sat = 1.0/(1.0 + A*f1)
    return f_lda + GAMMA*mphi**3*np.log1p((1.0 - g_sat)*E)


def pbc_c(rho, sigma, tau):
    """PBC correlation energy per particle (z=0, unpol)."""
    rs = (3.0/(4.0*np.pi*rho))**(1.0/3.0)
    xt2 = sigma/rho**(8.0/3.0)
    t_kin = tau/rho**(5.0/3.0)
    u = min(xt2/(8.0*t_kin), 1.0) if t_kin > 0.0 else 1.0
    t2_eff = 8.0*t_kin*u
    g = f_pbe_c(rs, 0.0, t2_eff)
    xs02 = 2.0**(2.0/3.0)*t2_eff
    br = (
        0.5*max(f_pbe_c(rs*2.0**(1.0/3.0), 1.0, xs02), g)
        + 0.5*max(f_pbe_c(rs*2.0**(1.0/3.0), -1.0, xs02), g)
    )
    f0 = (1.0 + 0.53*u**2)*g - (1.0 + 0.53)*u**2*br
    return f0*(1.0 + 2.8*f0*u**3)


if __name__ == "__main__":
    from pyscf import dft

    def ref(rho, sigma, tau):
        gx = np.sqrt(sigma) if sigma > 0 else 0.0
        raw = np.array([[rho], [gx], [0.0], [0.0], [tau]])
        return dft.libxc.eval_xc("MGGA_C_TPSS", raw)[0][0]

    # 1) равномерная плотность: PBC == PBE-c == PW92-modified LDA
    for rho, tau in ((0.7, 0.588), (0.1, 0.2), (5.0, 3.0)):
        d = abs(ref(rho, 0.0, tau) - pbc_c(rho, 0.0, tau))
        print(f"sigma=0 rho={rho}: diff = {d:.2e}")

    # 2) случайная сетка: умеренные точки
    rng = np.random.default_rng(2026)
    worst = 0.0; wpt = None
    for i in range(1500):
        r = 10**rng.uniform(-2.5, 1.0)
        t = 3.0*(r**2)*10**rng.uniform(-1.5, 1.5)   # физичный регион (t_kin ~ TF)
        u_raw = 10**rng.uniform(-2.5, 1.5)
        s = min(u_raw*8*r*t, 8*r*t*10**rng.uniform(0, 0.3))  # в основном U < 1
        d = abs(ref(r, s, t) - pbc_c(r, s, t))
        if d > worst:
            worst, wpt = d, (r, s, t)
    print(f"grid 1500 (умеренные): worst = {worst:.3e} at {wpt}")

    # 3) clamp-регион (U_raw > 1, sigma > 8*tau*rho^(1/3))
    worst = 0.0; wpt = None
    for i in range(500):
        r = 10**rng.uniform(-2.5, 1.0)
        t = 3.0*(r**2)*10**rng.uniform(-1.5, 1.5)
        s = 8*t*r**(1/3)*10**rng.uniform(0.0, 3.0)
        d = abs(ref(r, s, t) - pbc_c(r, s, t))
        if d > worst:
            worst, wpt = d, (r, s, t)
    print(f"grid 500 (clamp): worst = {worst:.3e} at {wpt}")

    # 4) малые плотности (до rho = 0.005)
    worst = 0.0; wpt = None
    for i in range(300):
        r = 10**rng.uniform(-2.5, -0.7)
        t = 3.0*(r**2)*10**rng.uniform(-1.0, 1.0)
        s = 8*r*t*10**rng.uniform(-1.0, 1.0)
        d = abs(ref(r, s, t) - pbc_c(r, s, t))
        if d > worst:
            worst, wpt = d, (r, s, t)
    print(f"grid 300 (low rho): worst = {worst:.3e} at {wpt}")
