"""Функционалы, заданные формулой от семи переменных (M06, M06-2X, TPSSh в UKS, ωB97X).

Производные даёт :mod:`quantumlab.engine.xc_ad`, формулы — :mod:`xc_meta` и
:mod:`xc_wb97`. Здесь — только оболочка: перевод между массивами SCF
(плотности, градиенты, ``τ``) и переменными ``(ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β)``
и обратно — в ``XcEvaluation`` / ``XcEvaluationSpin``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from quantumlab.engine.contracts import Array, XcEvaluation, XcEvaluationSpin
from quantumlab.engine.xc_ad import Dual
from quantumlab.engine.xc_meta import (
    CHANNEL_FLOOR,
    M06_2X_C,
    M06_2X_X,
    M06_C,
    M06_X,
    SpinInputs,
    make_inputs,
    minnesota_correlation,
    minnesota_exchange,
)
from quantumlab.engine.xc_wb97 import WB97X, WB97X_D, Wb97Parameters, wb97_semilocal

EnergyFormula = Callable[[SpinInputs], Dual]


class AdKernel:
    """Формула ``E_V(ρ_α, ρ_β, σ, τ)`` с выдачей энергии и первых производных."""

    def __init__(self, formula: EnergyFormula, *, uses_tau: bool) -> None:
        """``uses_tau`` — зависит ли формула от кинетической плотности."""
        self._formula = formula
        self.uses_tau = uses_tau

    def spin(
        self,
        density_spin: Array,
        density_gradient_spin: Array,
        tau_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Энергия и потенциалы по спин-каналам (UKS)."""
        rho = np.asarray(density_spin, dtype=float)
        grad = np.asarray(density_gradient_spin, dtype=float)
        s_aa = np.einsum("pd,pd->p", grad[0], grad[0])
        s_ab = np.einsum("pd,pd->p", grad[0], grad[1])
        s_bb = np.einsum("pd,pd->p", grad[1], grad[1])
        tau = None if tau_spin is None else np.asarray(tau_spin, dtype=float)
        if self.uses_tau and tau is None:
            msg = "Для meta-GGA нужна кинетическая плотность τ по каналам."
            raise ValueError(msg)
        inputs = make_inputs(
            rho[0],
            rho[1],
            s_aa,
            s_ab,
            s_bb,
            None if tau is None else tau[0],
            None if tau is None else tau[1],
        )
        energy = self._formula(inputs).finite()
        total = rho[0] + rho[1]
        valid = total > CHANNEL_FLOOR

        def partial(index: int) -> np.ndarray:
            return np.where(valid, energy.d.get(index, np.zeros_like(total)), 0.0)

        density = np.where(valid, energy.v / np.where(valid, total, 1.0), 0.0)
        vrho = np.stack([partial(0), partial(1)])
        cross = partial(3)
        vsigma = np.stack(
            [np.stack([partial(2), cross]), np.stack([cross, partial(4)])],
        )
        vtau = np.stack([partial(5), partial(6)]) if self.uses_tau else None
        return XcEvaluationSpin(energy_density=density, vrho=vrho, vsigma=vsigma, vtau=vtau)

    def closed_shell(
        self,
        density: Array,
        density_gradient: Array,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Замкнутая оболочка: ``ρ_α = ρ_β = ρ/2`` — через спиновую формулу."""
        rho = np.asarray(density, dtype=float)
        grad = np.asarray(density_gradient, dtype=float)
        half_tau = None if tau is None else np.stack([0.5 * tau, 0.5 * tau])
        spin = self.spin(
            np.stack([0.5 * rho, 0.5 * rho]), np.stack([0.5 * grad, 0.5 * grad]), half_tau
        )
        assert spin.vsigma is not None
        v_sigma = 0.25 * (spin.vsigma[0, 0] + 2.0 * spin.vsigma[0, 1] + spin.vsigma[1, 1])
        v_tau = None if spin.vtau is None else 0.5 * (spin.vtau[0] + spin.vtau[1])
        return XcEvaluation(
            energy_density=spin.energy_density,
            vrho=0.5 * (spin.vrho[0] + spin.vrho[1]),
            vsigma=v_sigma,
            vtau=v_tau,
        )


class _KernelFunctional:
    """Общая часть функционалов на ``AdKernel``."""

    name: str
    functional_class: str
    is_hybrid: bool = True
    exact_exchange_fraction: float
    #: Константа разделения ``ω`` оператора ``erf(ωr)/r``; ``0`` — функционал без разделения.
    range_separation_omega: float = 0.0
    #: Доля дальнодействующего (``erf``) точного обмена сверх ``exact_exchange_fraction``.
    long_range_exchange_fraction: float = 0.0
    requires_tau: bool
    _kernel: AdKernel

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Замкнутая оболочка."""
        del points
        if spin_polarized:
            msg = "Спин-поляризованное вычисление идёт через evaluate_spin."
            raise ValueError(msg)
        if density_gradient is None:
            msg = f"Функционалу «{self.name}» нужен градиент плотности."
            raise ValueError(msg)
        return self._kernel.closed_shell(density, density_gradient, tau)

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спин-разделённая версия для GGA (``requires_tau == False``)."""
        del points
        if self.requires_tau:
            msg = (
                f"Функционалу «{self.name}» нужна кинетическая плотность: "
                "используйте evaluate_spin_tau."
            )
            raise ValueError(msg)
        if density_gradient_spin is None:
            msg = f"Функционалу «{self.name}» нужен градиент плотности."
            raise ValueError(msg)
        return self._kernel.spin(density_spin, density_gradient_spin)

    def evaluate_spin_tau(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array,
        tau_spin: Array,
    ) -> XcEvaluationSpin:
        """Спин-разделённая версия meta-GGA: ``τ`` по каналам, ``(2, n_points)``."""
        del points
        return self._kernel.spin(density_spin, density_gradient_spin, tau_spin)


class M06(_KernelFunctional):
    """M06 (Zhao & Truhlar 2008): гибрид meta-GGA, 27 % точного обмена (LibXC 7.0.0)."""

    name = "m06"
    functional_class = "mgga"
    exact_exchange_fraction = 0.27
    requires_tau = True

    def __init__(self) -> None:
        """Собирает формулу из обмена и корреляции M06."""
        self._kernel = AdKernel(
            lambda inputs: minnesota_exchange(inputs, M06_X) + minnesota_correlation(inputs, M06_C),
            uses_tau=True,
        )


class M062x(_KernelFunctional):
    """M06-2X (Zhao & Truhlar 2008): гибрид meta-GGA, 54 % точного обмена (LibXC 7.0.0)."""

    name = "m062x"
    functional_class = "mgga"
    exact_exchange_fraction = 0.54
    requires_tau = True

    def __init__(self) -> None:
        """Собирает формулу из обмена и корреляции M06-2X."""
        self._kernel = AdKernel(
            lambda inputs: (
                minnesota_exchange(inputs, M06_2X_X) + minnesota_correlation(inputs, M06_2X_C)
            ),
            uses_tau=True,
        )


class _Wb97Family(_KernelFunctional):
    """Общая часть ωB97X и ωB97X-D: GGA с разделением ``ω`` и точным обменом."""

    functional_class = "range_separated_hybrid"
    requires_tau = False

    def __init__(self, parameters: Wb97Parameters) -> None:
        """Точный обмен: ``c_x`` на всём диапазоне и ``1 − c_x`` на дальнем (``erf``)."""
        self.exact_exchange_fraction = parameters.short_range_exact_exchange
        self.long_range_exchange_fraction = 1.0 - parameters.short_range_exact_exchange
        self.range_separation_omega = parameters.omega
        self._kernel = AdKernel(
            lambda inputs: wb97_semilocal(inputs, parameters),
            uses_tau=False,
        )


class Wb97x(_Wb97Family):
    """ωB97X (Chai & Head-Gordon 2008): ω = 0.3, ``c_x`` = 0.157706 (LibXC 7.0.0)."""

    name = "wb97x"

    def __init__(self) -> None:
        """Параметры ωB97X."""
        super().__init__(WB97X)


class Wb97xD(_Wb97Family):
    """ωB97X-D (Chai & Head-Gordon 2008): ω = 0.2, ``c_x`` = 0.222036.

    Полулокальная часть и точный обмен — как в LibXC (``HYB_GGA_XC_WB97X_D``);
    дисперсионная поправка D2 с CHG-затуханием считается отдельно
    (:mod:`quantumlab.engine.dispersion`) и подмешивается спецификацией задачи.
    """

    name = "wb97x-d"

    def __init__(self) -> None:
        """Параметры ωB97X-D."""
        super().__init__(WB97X_D)
