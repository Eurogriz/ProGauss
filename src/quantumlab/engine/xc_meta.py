"""Спин-поляризованные meta-GGA: TPSS, M06 и M06-2X (формулы LibXC 7.0.0).

Все формулы записаны от семи переменных ``(ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β)``
и возвращают **энергию на единицу объёма** ``E_V``; производные даёт
:mod:`quantumlab.engine.xc_ad`. Структура повторяет Maple-исходники LibXC
(``mgga_x_m06l.mpl``, ``hyb_mgga_x_m05.mpl``, ``mgga_c_m06l.mpl``,
``mgga_c_m05.mpl``, ``mgga_c_vsxc.mpl``, ``tpss_x.mpl``, ``tpss_c.mpl``,
``gga_c_pbe.mpl``, ``lda_c_pw.mpl``): сравнивать код со ссылкой строка к строке.

Соглашения LibXC, которые легко перепутать:

* ``τ_σ = ½ Σ_i |∇φ_iσ|²`` (с ½), безразмерные ``x_σ = |∇ρ_σ|/ρ_σ^{4/3}`` и
  ``t_σ = τ_σ/ρ_σ^{5/3}``;
* ``K = (3/10)(6π²)^{2/3}`` — значение ``t`` для однородного газа;
* корреляция Штолла: параллельная часть — ``ρ_σ·ε_c^{PW}(ρ_σ, 0)`` (полностью
  поляризованный газ плотности ``ρ_σ``), антипараллельная — остаток.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantumlab.engine import xc_ad as ad
from quantumlab.engine.xc_ad import Dual

#: Плотность канала, ниже которой его вклад считается нулевым (как порог плотности LibXC).
CHANNEL_FLOOR: float = 1e-14
_TAU_FLOOR: float = 1e-20
_RHO_MIN: float = 1e-30
_SIGMA_FLOOR: float = 1e-60
_ZETA_EPS: float = 2.2204460492503131e-16

#: C'_x = (3/8)·2^{4/3}·(3/π)^{1/3}: ``e_x = −C'_x ρ_σ^{4/3}`` для спин-канала.
X_FACTOR_C: float = (3.0 / 8.0) * 2.0 ** (4.0 / 3.0) * (3.0 / math.pi) ** (1.0 / 3.0)
K_FACTOR_C: float = (3.0 / 10.0) * (6.0 * math.pi**2) ** (2.0 / 3.0)
X2S: float = 1.0 / (2.0 * (6.0 * math.pi**2) ** (1.0 / 3.0))
MU_GE: float = 10.0 / 81.0
RS_FACTOR: float = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

# -- PBE-обмен (gga_x_pbe.mpl) ------------------------------------------------
PBE_KAPPA: float = 0.8040
PBE_MU: float = 0.2195149727645171

# -- PW92 modified (lda_c_pw.mpl, lda_c_pw_modified_params) ---------------------
_PW_A = (0.0310907, 0.01554535, 0.0168869)
#: Исходная параметризация PW92 (``lda_c_pw_params``): ωB97X и B97 берут её, а не «modified».
_PW_A_ORIGINAL = (0.031091, 0.015545, 0.016887)
_PW_FZ20_ORIGINAL: float = 1.709921
_PW_ALPHA1 = (0.21370, 0.20548, 0.11125)
_PW_BETA1 = (7.5957, 14.1189, 10.357)
_PW_BETA2 = (3.5876, 6.1977, 3.6231)
_PW_BETA3 = (1.6382, 3.3662, 0.88026)
_PW_BETA4 = (0.49294, 0.62517, 0.49671)
_PW_FZ20: float = 1.709920934161365617563962776245


@dataclass(frozen=True, slots=True)
class SpinInputs:
    """Семь независимых переменных в виде ``Dual`` и маски активных каналов."""

    ra: Dual
    rb: Dual
    saa: Dual
    sab: Dual
    sbb: Dual
    ta: Dual
    tb: Dual
    active_a: np.ndarray
    active_b: np.ndarray
    active: np.ndarray


def make_inputs(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    s_aa: np.ndarray,
    s_ab: np.ndarray,
    s_bb: np.ndarray,
    tau_a: np.ndarray | None = None,
    tau_b: np.ndarray | None = None,
) -> SpinInputs:
    """Подготавливает переменные: клампы как в LibXC, «мёртвые» точки — безопасными числами."""
    ra = np.asarray(rho_a, dtype=float)
    rb = np.asarray(rho_b, dtype=float)
    active_a = ra > CHANNEL_FLOOR
    active_b = rb > CHANNEL_FLOOR
    active = (ra + rb) > CHANNEL_FLOOR
    # Неактивный канал не подменяется «безопасной» единицей: полная плотность
    # входит в корреляцию (ζ, r_s), и подмена исказила бы вклад полностью
    # поляризованных областей. Как в LibXC, значения только поджимаются к малому
    # порогу — формулы остаются конечными, а вклад самого канала маскируется.
    ra_safe = np.maximum(ra, _RHO_MIN)
    rb_safe = np.maximum(rb, _RHO_MIN)
    zeros = np.zeros_like(ra)
    ta = zeros if tau_a is None else np.asarray(tau_a, dtype=float)
    tb = zeros if tau_b is None else np.asarray(tau_b, dtype=float)
    ta_safe = np.maximum(ta, _TAU_FLOOR)
    tb_safe = np.maximum(tb, _TAU_FLOOR)
    saa = np.maximum(np.asarray(s_aa, dtype=float), _SIGMA_FLOOR)
    sbb = np.maximum(np.asarray(s_bb, dtype=float), _SIGMA_FLOOR)
    # Предел фон Вайцзекера для каждого канала (σ_σσ ≤ 8 ρ_σ τ_σ), как в work_mgga LibXC.
    if tau_a is not None:
        saa = np.minimum(saa, 8.0 * ra_safe * ta_safe)
        sbb = np.minimum(sbb, 8.0 * rb_safe * tb_safe)
    # Неравенство Коши—Буняковского |σ_αβ| ≤ √(σ_αα σ_ββ) после клампов сохраняется.
    bound = np.sqrt(saa * sbb)
    sab = np.clip(np.asarray(s_ab, dtype=float), -bound, bound)
    return SpinInputs(
        ra=Dual.variable(0, ra_safe),
        rb=Dual.variable(1, rb_safe),
        saa=Dual.variable(2, saa),
        sab=Dual.variable(3, sab),
        sbb=Dual.variable(4, sbb),
        ta=Dual.variable(5, ta_safe),
        tb=Dual.variable(6, tb_safe),
        active_a=active_a,
        active_b=active_b,
        active=active,
    )


# -- общие строительные блоки ---------------------------------------------------
def pbe_enhancement(x: Dual) -> Dual:
    """``F_x^{PBE}(x)`` по ``s = X2S·x``."""
    s2 = (x * x) * (X2S * X2S)
    return 1.0 + PBE_KAPPA * (1.0 - PBE_KAPPA / (PBE_KAPPA + PBE_MU * s2))


def _pw_g(k: int, rs: Dual, original: bool = False) -> Dual:
    """``g(k, rs)`` из PW92 (−α при k = 2); ``original`` — исходные (не modified) ``A``."""
    a = (_PW_A_ORIGINAL if original else _PW_A)[k]
    sqrt_rs = ad.sqrt(rs)
    aux = (
        sqrt_rs * _PW_BETA1[k]
        + rs * _PW_BETA2[k]
        + (rs * sqrt_rs) * _PW_BETA3[k]
        + (rs * rs) * _PW_BETA4[k]
    )
    return (1.0 + rs * _PW_ALPHA1[k]) * ad.log1p(1.0 / (aux * (2.0 * a))) * (-2.0 * a)


def f_zeta(zeta: Dual) -> Dual:
    """``f(ζ)`` — интерполяция спиновой поляризации PW92."""
    return (((1.0 + zeta) ** (4.0 / 3.0)) + ((1.0 - zeta) ** (4.0 / 3.0)) - 2.0) / (
        float(2.0 ** (4.0 / 3.0)) - 2.0
    )


def pw92(rs: Dual, zeta: Dual, original: bool = False) -> Dual:
    """``ε_c^{PW92}(r_s, ζ)`` на частицу (``mod`` по умолчанию, ``original`` — как в B97)."""
    g1, g2, g3 = _pw_g(0, rs, original), _pw_g(1, rs, original), _pw_g(2, rs, original)
    fz20 = _PW_FZ20_ORIGINAL if original else _PW_FZ20
    fz = f_zeta(zeta)
    z4 = (zeta * zeta) * (zeta * zeta)
    return g1 + z4 * fz * (g2 - g1 + g3 / fz20) - fz * g3 / fz20


def pw92_polarized(rs: Dual, original: bool = False) -> Dual:
    """``ε_c^{PW92}(r_s, ζ = 1) = g(2, r_s)`` — точно, без ``f(1)``."""
    return _pw_g(1, rs, original)


def rs_of(rho: Dual) -> Dual:
    """Радиус Вигнера—Зейтца плотности ``rho``."""
    return (rho ** (-1.0 / 3.0)) * RS_FACTOR


def stoll_parallel(rho_sigma: Dual, original: bool = False) -> Dual:
    """``ρ_σ ε_c^{PW}(ρ_σ, 0)`` — параллельная корреляция Штолла на единицу объёма."""
    return rho_sigma * pw92_polarized(rs_of(rho_sigma), original)


def stoll_perpendicular(inputs: SpinInputs, original: bool = False) -> Dual:
    """Антипараллельная часть: ``ρ ε_c(ρ_α, ρ_β) − ρ_α ε_c(ρ_α, 0) − ρ_β ε_c(0, ρ_β)``."""
    rho = inputs.ra + inputs.rb
    zeta = clamp_zeta((inputs.ra - inputs.rb) / rho)
    total = rho * pw92(rs_of(rho), zeta, original)
    par_a = stoll_parallel(inputs.ra, original).masked(inputs.active_a)
    par_b = stoll_parallel(inputs.rb, original).masked(inputs.active_b)
    return total - par_a - par_b


def clamp_zeta(zeta: Dual) -> Dual:
    """Кламп ζ на границах, как ``z_thr`` LibXC (производная — от исходной формулы)."""
    value = zeta.v
    value = np.where(1.0 + value < _ZETA_EPS, -1.0 + _ZETA_EPS, value)
    value = np.where(1.0 - value < _ZETA_EPS, 1.0 - _ZETA_EPS, value)
    return Dual(value, zeta.d)


def reduced_gradient(sigma: Dual, rho: Dual) -> Dual:
    """``x = √σ/ρ^{4/3}``."""
    return ad.sqrt(sigma) * (rho ** (-4.0 / 3.0))


def reduced_tau(tau: Dual, rho: Dual) -> Dual:
    """``t = τ/ρ^{5/3}``."""
    return tau * (rho ** (-5.0 / 3.0))


def mgga_w(t: Dual) -> Dual:
    """``w = (K − t)/(K + t)``."""
    return (K_FACTOR_C - t) / (K_FACTOR_C + t)


def mgga_series_w(coefficients: tuple[float, ...], t: Dual) -> Dual:
    """``Σ a_i w^i`` — схема Хорнера по ``w``."""
    w = mgga_w(t)
    result = as_const_like(coefficients[-1], w)
    for coefficient in reversed(coefficients[:-1]):
        result = result * w + coefficient
    return result


def as_const_like(value: float, like: Dual) -> Dual:
    """Константа с формой ``like`` (для начала схемы Хорнера)."""
    return Dual(np.full_like(like.v, value))


def gtv4(alpha: float, d: tuple[float, ...], x: Dual, z: Dual) -> Dual:
    """``gtv4`` из ``gvt4.mpl`` (VS98-член)."""
    gamma = 1.0 + (x * x + z) * alpha
    inv = gamma.reciprocal()
    inv2 = inv * inv
    inv3 = inv2 * inv
    x2 = x * x
    return (
        inv * d[0]
        + (x2 * d[1] + z * d[2]) * inv2
        + (x2 * x2 * d[3] + x2 * z * d[4] + z * z * d[5]) * inv3
    )


# -- обмен M05/M06 ---------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MinnesotaExchange:
    """Параметры обмена M05/M06: ``F = csi·F_PBE·Σ a_i w^i + gtv4(α, d, x, 2(t−K))``."""

    a: tuple[float, ...]
    d: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    csi: float = 1.0
    alpha: float = 0.00186726


M06_X = MinnesotaExchange(
    a=(
        5.877943e-01,
        -1.371776e-01,
        2.682367e-01,
        -2.515898e00,
        -2.978892e00,
        8.710679e00,
        1.688195e01,
        -4.489724e00,
        -3.299983e01,
        -1.449050e01,
        2.043747e01,
        1.256504e01,
    ),
    d=(1.422057e-01, 7.370319e-04, -1.601373e-02, 0.0, 0.0, 0.0),
)
M06_2X_X = MinnesotaExchange(
    a=(
        4.600000e-01,
        -2.206052e-01,
        -9.431788e-02,
        2.164494e00,
        -2.556466e00,
        -1.422133e01,
        1.555044e01,
        3.598078e01,
        -2.722754e01,
        -3.924093e01,
        1.522808e01,
        1.522227e01,
    ),
)


def minnesota_exchange(inputs: SpinInputs, p: MinnesotaExchange) -> Dual:
    """Обмен M05/M06 на единицу объёма: ``Σ_σ −C'_x ρ_σ^{4/3} F(x_σ, t_σ)``."""
    total: Dual | None = None
    for rho, sigma, tau, active in (
        (inputs.ra, inputs.saa, inputs.ta, inputs.active_a),
        (inputs.rb, inputs.sbb, inputs.tb, inputs.active_b),
    ):
        x = reduced_gradient(sigma, rho)
        t = reduced_tau(tau, rho)
        enhancement = pbe_enhancement(x) * mgga_series_w(p.a, t) * p.csi
        if any(p.d):
            enhancement = enhancement + gtv4(p.alpha, p.d, x, (t - K_FACTOR_C) * 2.0)
        term = ((rho ** (4.0 / 3.0)) * enhancement * (-X_FACTOR_C)).masked(active)
        total = term if total is None else total + term
    assert total is not None
    return total


# -- корреляция M06 ---------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MinnesotaCorrelation:
    """Параметры корреляции M06: B97-ряд с поправкой Ферми плюс VS98-член."""

    css: tuple[float, ...]
    cab: tuple[float, ...]
    dss: tuple[float, ...]
    dab: tuple[float, ...]
    gamma_ss: float = 0.06
    gamma_ab: float = 0.0031
    alpha_ss: float = 0.00515088
    alpha_ab: float = 0.00304966
    fermi_d_cnst: float = 1e-10


M06_C = MinnesotaCorrelation(
    css=(5.094055e-01, -1.491085e00, 1.723922e01, -3.859018e01, 2.845044e01),
    cab=(3.741539e00, 2.187098e02, -4.531252e02, 2.936479e02, -6.287470e01),
    dss=(4.905945e-01, -1.437348e-01, 2.357824e-01, 1.871015e-03, -3.788963e-03, 0.0),
    dab=(-2.741539e00, -6.720113e-01, -7.932688e-02, 1.918681e-03, -2.032902e-03, 0.0),
)
M06_2X_C = MinnesotaCorrelation(
    css=(3.097855e-01, -5.528642e00, 1.347420e01, -3.213623e01, 2.846742e01),
    cab=(8.833596e-01, 3.357972e01, -7.043548e01, 4.978271e01, -1.852891e01),
    dss=(6.902145e-01, 9.847204e-02, 2.214797e-01, -1.968264e-03, -6.775479e-03, 0.0),
    dab=(1.166404e-01, -9.120847e-02, -6.726189e-02, 6.720580e-05, 8.448011e-04, 0.0),
)


def b97_series(gamma: float, coefficients: tuple[float, ...], x: Dual) -> Dual:
    """``Σ c_i (γx²/(1+γx²))^{i−1}``."""
    gx2 = (x * x) * gamma
    u = gx2 / (1.0 + gx2)
    result = as_const_like(coefficients[-1], u)
    for coefficient in reversed(coefficients[:-1]):
        result = result * u + coefficient
    return result


def minnesota_correlation(inputs: SpinInputs, p: MinnesotaCorrelation) -> Dual:
    """Корреляция M06 на единицу объёма."""
    total: Dual | None = None
    channels = (
        (inputs.ra, inputs.saa, inputs.ta, inputs.active_a),
        (inputs.rb, inputs.sbb, inputs.tb, inputs.active_b),
    )
    x_channels: list[Dual] = []
    t_channels: list[Dual] = []
    for rho, sigma, tau, active in channels:
        x = reduced_gradient(sigma, rho)
        t = reduced_tau(tau, rho)
        x_channels.append(x)
        t_channels.append(t)
        fermi_d = 1.0 - (x * x) / (t * 8.0)
        damping = 1.0 - ad.exp((t * t) * (-4.0 / p.fermi_d_cnst**2))
        same_spin = (
            b97_series(p.gamma_ss, p.css, x) * fermi_d * damping
            + gtv4(p.alpha_ss, p.dss, x, (t - K_FACTOR_C) * 2.0) * fermi_d
        )
        term = (stoll_parallel(rho) * same_spin).masked(active)
        total = term if total is None else total + term
    assert total is not None
    x_ab = ad.sqrt(x_channels[0] * x_channels[0] + x_channels[1] * x_channels[1])
    opposite = b97_series(p.gamma_ab, p.cab, x_ab) + gtv4(
        p.alpha_ab,
        p.dab,
        x_ab,
        (t_channels[0] + t_channels[1] - 2.0 * K_FACTOR_C) * 2.0,
    )
    both = inputs.active_a & inputs.active_b
    return total + (stoll_perpendicular(inputs) * opposite).masked(both)


# -- TPSS --------------------------------------------------------------------------
_TPSS_X_B: float = 0.40
_TPSS_X_C: float = 1.59096
_TPSS_X_E: float = 1.537
_TPSS_X_KAPPA: float = 0.804
_TPSS_X_MU: float = 0.21951
_TPSS_BLOC_A: float = 2.0
_TPSS_BLOC_B: float = 0.0


def tpss_enhancement(x: Dual, t: Dual) -> Dual:
    """``F_x^{TPSS}(x, t)`` (``tpss_x.mpl`` вместе с уравнением (5))."""
    p = (x * x) * (X2S * X2S)
    z = (x * x) / (t * 8.0)
    alpha = (t - (x * x) / 8.0) / K_FACTOR_C
    qb = (alpha - 1.0) * 0.45 / ad.sqrt(1.0 + alpha * (alpha - 1.0) * _TPSS_X_B) + p * (2.0 / 3.0)
    z_power = ad.exp(ad.log(z) * (_TPSS_BLOC_A + z * _TPSS_BLOC_B))
    numerator = (
        (MU_GE + z_power * _TPSS_X_C / ((z * z + 1.0) ** 2.0)) * p
        + (qb * qb) * (146.0 / 2025.0)
        - qb * ad.sqrt((z * z * (9.0 / 25.0) + p * p) * 0.5) * (73.0 / 405.0)
        + (p * p) * (MU_GE**2 / _TPSS_X_KAPPA)
        + (z * z) * (2.0 * math.sqrt(_TPSS_X_E) * MU_GE * 9.0 / 25.0)
        + (p * p * p) * (_TPSS_X_E * _TPSS_X_MU)
    )
    denominator = (1.0 + p * math.sqrt(_TPSS_X_E)) ** 2.0
    fx = numerator / denominator
    a1 = _TPSS_X_KAPPA / (fx + _TPSS_X_KAPPA)
    return 1.0 + (1.0 - a1) * _TPSS_X_KAPPA


def tpss_exchange(inputs: SpinInputs) -> Dual:
    """Обмен TPSS на единицу объёма (спиновое масштабирование)."""
    total: Dual | None = None
    for rho, sigma, tau, active in (
        (inputs.ra, inputs.saa, inputs.ta, inputs.active_a),
        (inputs.rb, inputs.sbb, inputs.tb, inputs.active_b),
    ):
        x = reduced_gradient(sigma, rho)
        t = reduced_tau(tau, rho)
        term = ((rho ** (4.0 / 3.0)) * tpss_enhancement(x, t) * (-X_FACTOR_C)).masked(active)
        total = term if total is None else total + term
    assert total is not None
    return total


_PBC_GAMMA: float = (1.0 - math.log(2.0)) / math.pi**2
_PBC_BETA: float = 0.06672455060314922
_TPSS_C0 = (0.53, 0.87, 0.50, 2.26)
_TPSS_D: float = 2.8


def _pbe_correlation_per_particle(rs: Dual, zeta: Dual | None, xt: Dual) -> Dual:
    """``ε_c^{PBE}(r_s, ζ, x_t)``; ``zeta=None`` — полностью поляризованный газ (ζ = 1)."""
    if zeta is None:
        eps = pw92_polarized(rs)
        phi_value = 2.0 ** (-1.0 / 3.0)
        t = xt / (ad.sqrt(rs) * 4.0)
        phi3 = as_const_like(phi_value * phi_value * phi_value, rs)
    else:
        eps = pw92(rs, zeta)
        phi = (((1.0 + zeta) ** (2.0 / 3.0)) + ((1.0 - zeta) ** (2.0 / 3.0))) * 0.5
        t = xt / (phi * ad.sqrt(rs) * (4.0 * 2.0 ** (1.0 / 3.0)))
        phi3 = phi * phi * phi
    a_factor = _PBC_BETA / (_PBC_GAMMA * (ad.exp(eps / (phi3 * (-_PBC_GAMMA))) - 1.0))
    t2 = t * t
    f1 = t2 + a_factor * t2 * t2
    f2 = f1 * _PBC_BETA / ((1.0 + a_factor * f1) * _PBC_GAMMA)
    return eps + phi3 * ad.log1p(f2) * _PBC_GAMMA


def tpss_correlation(inputs: SpinInputs) -> Dual:
    """Корреляция TPSS (``mgga_c_tpss``) на единицу объёма."""
    rho = inputs.ra + inputs.rb
    zeta = clamp_zeta((inputs.ra - inputs.rb) / rho)
    rs = rs_of(rho)
    sigma_total = inputs.saa + inputs.sab * 2.0 + inputs.sbb
    xt = ad.sqrt(sigma_total) * (rho ** (-4.0 / 3.0))
    xs0 = reduced_gradient(inputs.saa, inputs.ra)
    xs1 = reduced_gradient(inputs.sbb, inputs.rb)
    one_p = (1.0 + zeta) * 0.5
    one_m = (1.0 - zeta) * 0.5

    def t_total(a: Dual, b: Dual) -> Dual:
        return a * (one_p ** (5.0 / 3.0)) + b * (one_m ** (5.0 / 3.0))

    ts0 = reduced_tau(inputs.ta, inputs.ra)
    ts1 = reduced_tau(inputs.tb, inputs.rb)
    aux = ad.minimum(xt * xt / (t_total(ts0, ts1) * 8.0), 1.0)

    xi2 = (
        (1.0 - zeta * zeta)
        * (t_total(xs0 * xs0, xs1 * xs1) - xt * xt)
        / (2.0 * (3.0 * math.pi**2) ** (1.0 / 3.0)) ** 2
    )
    z2 = zeta * zeta
    c00 = z2 * (z2 * (z2 * _TPSS_C0[3] + _TPSS_C0[2]) + _TPSS_C0[1]) + _TPSS_C0[0]
    c0_den = 1.0 + xi2 * (((1.0 + zeta) ** (-4.0 / 3.0)) + ((1.0 - zeta) ** (-4.0 / 3.0))) * 0.5
    c0_regular = c00 / (c0_den**4.0)
    c0 = ad.where(1.0 - np.abs(zeta.v) <= 1e-12, sum(_TPSS_C0), c0_regular)

    eps_total = _pbe_correlation_per_particle(rs, zeta, xt)
    eps_a_own = _pbe_correlation_per_particle(
        rs_of(inputs.ra), None, xs0
    )  # полностью поляризованный газ плотности ρ_α
    eps_b_own = _pbe_correlation_per_particle(rs_of(inputs.rb), None, xs1)
    eps_a = ad.where(inputs.active_a, ad.maximum(eps_a_own, eps_total), eps_total)
    eps_b = ad.where(inputs.active_b, ad.maximum(eps_b_own, eps_total), eps_total)

    aux2 = aux * aux
    parallel = (c0 + 1.0) * aux2 * (eps_a * one_p + eps_b * one_m) * (-1.0)
    perpendicular = (c0 * aux2 + 1.0) * eps_total
    f0 = parallel + perpendicular
    energy_per_particle = f0 * (1.0 + f0 * aux2 * aux * _TPSS_D)
    return rho * energy_per_particle


def tpss_xc(inputs: SpinInputs, exchange_scale: float = 1.0) -> Dual:
    """``exchange_scale · E_x^{TPSS} + E_c^{TPSS}``."""
    return tpss_exchange(inputs) * exchange_scale + tpss_correlation(inputs)
