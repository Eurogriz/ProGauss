"""Формулы ωB97X и ωB97X-D (Chai, Head-Gordon 2008) на автоматическом дифференцировании.

Функционал::

    E_xc = E_x^{LR-HF} + c_x E_x^{SR-HF} + E_x^{SR-B97} + E_c^{B97},

где короткодействующий обмен B97 — это LDA-обмен с экранировкой ``erfc(ωr)/r``
(Toulouse; Tawada et al.), умноженный на ряд по ``u = γx²/(1+γx²)``. Здесь —
только полулокальная часть ``E_x^{SR-B97} + E_c^{B97}``: точный обмен
(дальнодействующий ``erf`` и доля ``c_x`` короткодействующего) строится в
SCF-слое по интегралам с ``erf(ωr)/r``.

Формулы следуют ``hyb_gga_xc_wb97.mpl``, ``b97.mpl``, ``lda_x_erf.mpl``,
``attenuation.mpl`` (LibXC 7.0.0); корреляция — PW92 в исходной
(немодифицированной) параметризации.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantumlab.engine import xc_ad as ad
from quantumlab.engine.xc_ad import Dual
from quantumlab.engine.xc_meta import (
    RS_FACTOR,
    X_FACTOR_C,
    SpinInputs,
    b97_series,
    reduced_gradient,
    stoll_parallel,
    stoll_perpendicular,
)

#: Порог переключения на асимптотический ряд (``enforce_smooth_lr(…, 1.35, 16)`` LibXC).
_ATTENUATION_CUTOFF: float = 1.35

#: Коэффициенты ряда ``F(a) = Σ c_k a^{−2k}`` при больших ``a`` (``k = 1…12``).
#: Выведены разложением полной формулы по ``y = 1/(2a)`` в точной рациональной
#: арифметике; на границе ряд и формула совпадают до ≈1e-16.
_ATTENUATION_SERIES: tuple[float, ...] = (
    0.027777777777777776,
    -0.0010416666666666667,
    3.7202380952380956e-05,
    -1.2056327160493827e-06,
    3.522952741702742e-08,
    -9.315500038156288e-10,
    2.2426203795561436e-11,
    -4.94695671960914e-13,
    1.0059600984851122e-14,
    -1.8961549475413822e-16,
    3.3293690550475775e-18,
    -5.469677733292448e-20,
)


def _clip(x: Dual, low: float, high: float) -> Dual:
    """Значение поджимается в ``[low, high]`` (производные остаются от аргумента)."""
    return Dual(np.clip(x.v, low, high), x.d)


def attenuation_erf(a: Dual) -> Dual:
    """``F(a) = 1 − (8a/3)[√π erf(1/2a) + 2a(e^{−1/4a²} − 1 − 2a²(e^{−1/4a²} − 1) − ½)]``.

    Множитель короткодействующего (``erfc``) LDA-обмена при ``a = ω/(2k_σ)``.
    Полная формула неустойчива при больших ``a`` (вычитание величин порядка
    ``a³``), поэтому выше ``a = 1.35`` берётся ряд по ``1/a²``.
    """
    low = _clip(a, 1e-8, _ATTENUATION_CUTOFF)
    exponent = ad.exp(-1.0 / (low * low * 4.0))
    aux2 = exponent - 1.0
    aux3 = (low * low) * aux2 * 2.0 + 0.5
    full = 1.0 - (low * 8.0 / 3.0) * (
        ad.erf(1.0 / (low * 2.0)) * math.sqrt(math.pi) + low * 2.0 * (aux2 - aux3)
    )
    high = _clip(a, _ATTENUATION_CUTOFF, 1e300)
    z = 1.0 / (high * high)
    series = ad.as_dual(_ATTENUATION_SERIES[-1]) * z
    for coefficient in reversed(_ATTENUATION_SERIES[:-1]):
        series = (series + coefficient) * z
    return ad.where(a.v < _ATTENUATION_CUTOFF, full, series)


def short_range_lda_exchange(rho_sigma: Dual, omega: float) -> Dual:
    """``e_xσ^{SR-LDA} = −C'_x ρ_σ^{4/3} F(ω/(2k_σ))`` на единицу объёма."""
    k_f = (rho_sigma * (6.0 * math.pi**2)) ** (1.0 / 3.0)
    return -(rho_sigma ** (4.0 / 3.0)) * X_FACTOR_C * attenuation_erf(omega / (k_f * 2.0))


@dataclass(frozen=True, slots=True)
class Wb97Parameters:
    """Параметры семейства ωB97: ряды B97 и константа разделения ``ω``."""

    omega: float
    short_range_exact_exchange: float
    cx: tuple[float, ...]
    css: tuple[float, ...]
    cos: tuple[float, ...]


WB97X = Wb97Parameters(
    omega=0.3,
    short_range_exact_exchange=1.57706e-01,
    cx=(8.42294e-01, 7.26479e-01, 1.04760e00, -5.70635e00, 1.32794e01),
    css=(1.00000e00, -4.33879e00, 1.82308e01, -3.17430e01, 1.72901e01),
    cos=(1.00000e00, 2.37031e00, -1.13995e01, 6.58405e00, -3.78132e00),
)

WB97X_D = Wb97Parameters(
    omega=0.2,
    short_range_exact_exchange=2.22036e-01,
    cx=(7.77964e-01, 6.61160e-01, 5.74541e-01, -5.25671e00, 1.16386e01),
    css=(1.00000e00, -6.90539e00, 3.13343e01, -5.10533e01, 2.64423e01),
    cos=(1.00000e00, 1.79413e00, -1.20477e01, 1.40847e01, -8.50809e00),
)

_GAMMA_X: float = 0.004
_GAMMA_SS: float = 0.2
_GAMMA_OS: float = 0.006


def wb97_semilocal(inputs: SpinInputs, p: Wb97Parameters) -> Dual:
    """``E_x^{SR-B97} + E_c^{B97}`` на единицу объёма (спиновая формула)."""
    total: Dual | None = None
    channels = (
        (inputs.ra, inputs.saa, inputs.active_a),
        (inputs.rb, inputs.sbb, inputs.active_b),
    )
    x_channels: list[Dual] = []
    for rho, sigma, active in channels:
        x = reduced_gradient(sigma, rho)
        x_channels.append(x)
        exchange = short_range_lda_exchange(rho, p.omega) * b97_series(_GAMMA_X, p.cx, x)
        same_spin = stoll_parallel(rho, True) * b97_series(_GAMMA_SS, p.css, x)
        term = (exchange + same_spin).masked(active)
        total = term if total is None else total + term
    assert total is not None
    x_average = ad.sqrt((x_channels[0] * x_channels[0] + x_channels[1] * x_channels[1]) * 0.5)
    both = inputs.active_a & inputs.active_b
    opposite = stoll_perpendicular(inputs, True) * b97_series(_GAMMA_OS, p.cos, x_average)
    return total + opposite.masked(both)


__all__ = [
    "RS_FACTOR",
    "WB97X",
    "WB97X_D",
    "Wb97Parameters",
    "attenuation_erf",
    "short_range_lda_exchange",
    "wb97_semilocal",
]
