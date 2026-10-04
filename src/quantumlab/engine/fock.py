"""Построитель спиновых фокианов: ``(Dα, Dβ) → (Fα, Fβ, E)`` для HF и DFT.

Решатели (``run_uhf``, ``run_uks``) сворачивают построение фокиана в свой
итерационный цикл. Анализу устойчивости нужен тот же оператор как отдельная
функция: гессиан по вращениям орбиталей получается конечной разностью
орбитального градиента, а градиент — это фокиан в точке, смещённой от
сошедшегося решения. Поэтому здесь один спин-неограниченный оператор, который
годится и для RHF/RKS (``Dα = Dβ``), и для ROHF (общие орбитали, разные
занятия), и для комплексных орбиталей.

Комплексная плотность
---------------------
Кулоновский и обменный члены линейны по ``D`` и поэтому определены для
комплексной эрмитовой матрицы. Обменно-корреляционный функционал зависит от
плотности электронов ``ρ(r) = Σ|ψ_i(r)|²``, то есть от **вещественной** части
``Re D``; мнимая антисимметричная часть ни ``ρ``, ни ``∇ρ`` не меняет. Так
определена энергия комплексного детерминанта Кона–Шэма, и именно её градиент
нужен для мнимых вращений.
"""

from __future__ import annotations

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import BasisSet, nuclear_repulsion
from quantumlab.engine.contracts import ExchangeCorrelationFunctional, range_separation
from quantumlab.engine.dft import xc_matrix_and_energy_spin
from quantumlab.engine.functional import (
    density_at_points,
    density_gradient_at_points,
    evaluate_basis_with_gradients,
    kinetic_density_at_points,
)
from quantumlab.engine.quadrature import QuadratureGrid, build_grid
from quantumlab.engine.scf import (
    ExactExchange,
    PrecomputedIntegrals,
    build_integrals,
    coulomb_matrix,
    exchange_matrix,
)


class SpinFockBuilder:
    """Спин-неограниченный оператор Фока и энергия для HF или DFT.

    ``functional is None`` — чистый Хартри–Фок (доля обмена 1). Иначе —
    Кон–Шэм со своей долей точного обмена и спин-поляризованным XC; для
    meta-GGA в XC входит кинетическая плотность каждого канала.
    """

    def __init__(
        self,
        basis: BasisSet,
        molecule: Molecule,
        *,
        functional: ExchangeCorrelationFunctional | None = None,
        integrals: PrecomputedIntegrals | None = None,
        grid: QuadratureGrid | None = None,
    ) -> None:
        """Готовит интегралы и (для DFT) базис на сетке; плотности подаются позже."""
        omega = range_separation(functional)[0] if functional is not None else 0.0
        prepared = (
            integrals if integrals is not None else build_integrals(basis, molecule, omega=omega)
        )
        self.core = prepared.core
        self.overlap = prepared.overlap
        self.eri = prepared.eri
        self.exchange = (
            ExactExchange.for_functional(functional, prepared) if functional is not None else None
        )
        self.nuclear_repulsion = nuclear_repulsion(molecule)
        self.functional = functional
        self._grid: QuadratureGrid | None = None
        self._values: np.ndarray | None = None
        self._gradients: np.ndarray | None = None
        if functional is not None:
            self._grid = grid if grid is not None else build_grid(molecule)
            self._values, self._gradients = evaluate_basis_with_gradients(
                basis, molecule, self._grid.points
            )

    def __call__(
        self, density_alpha: np.ndarray, density_beta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Фокианы каналов и полная энергия (с ядерным отталкиванием).

        Для комплексной эрмитовой плотности фокианы тоже комплексны и эрмитовы,
        энергия вещественна.
        """
        total = density_alpha + density_beta
        coulomb = coulomb_matrix(total, self.eri)
        if self.exchange is None:
            # Чистый Хартри–Фок: весь обмен точный, без разделения.
            exchange_alpha = exchange_matrix(density_alpha, self.eri)
            exchange_beta = exchange_matrix(density_beta, self.eri)
        else:
            # Уже взвешенный оператор c·K + c_lr·K_erf (для чистого GGA — нули).
            exchange_alpha = self.exchange(density_alpha)
            exchange_beta = self.exchange(density_beta)

        energy = float(
            np.real(np.sum(total * self.core) + 0.5 * np.sum(total * coulomb))
            - 0.5
            * np.real(np.sum(density_alpha * exchange_alpha) + np.sum(density_beta * exchange_beta))
            + self.nuclear_repulsion
        )
        fock_alpha = self.core + coulomb - exchange_alpha
        fock_beta = self.core + coulomb - exchange_beta

        if self.functional is not None:
            assert self._grid is not None
            assert self._values is not None
            assert self._gradients is not None
            real_alpha = np.real(density_alpha)
            real_beta = np.real(density_beta)
            tau_alpha = tau_beta = None
            if self.functional.requires_tau:
                tau_alpha = kinetic_density_at_points(self._gradients, real_alpha)
                tau_beta = kinetic_density_at_points(self._gradients, real_beta)
            v_alpha, v_beta, xc_energy = xc_matrix_and_energy_spin(
                self._grid,
                self._values,
                self._gradients,
                density_at_points(self._values, real_alpha),
                density_at_points(self._values, real_beta),
                density_gradient_at_points(self._values, self._gradients, real_alpha),
                density_gradient_at_points(self._values, self._gradients, real_beta),
                self.functional,
                tau_alpha,
                tau_beta,
            )
            fock_alpha = fock_alpha + v_alpha
            fock_beta = fock_beta + v_beta
            energy += xc_energy
        return fock_alpha, fock_beta, energy
