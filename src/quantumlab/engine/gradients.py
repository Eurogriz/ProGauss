"""Аналитические градиенты RHF по декартовым координатам ядер.

Формула (Helgaker, Jørgensen, Olsen, гл. 10) для энергии
``E = Σ D·h + ½ ΣΣ D·D·[(μν|λσ) − ½(μλ|νσ)] + E_nuc``::

    dE/dA_x = Σ_μν D_μν h^x_μν
            + ½ ΣΣ D_μν D_λσ (μν|λσ)^x
            − ¼ ΣΣ D_μν D_λσ (μλ|νσ)^x
            − Σ_μν W_μν S^x_μν
            + dE_nuc/dA_x

где ``W_μν = 2Σ_i^{занято} C_μi ε_i C_νi`` — энерговзвешенная плотность, а
штрих ``x`` означает полную производную по координате ядра.

Три источника геометрической зависимости, и потерять любой из них — значит
получить неверный градиент при правдоподобной энергии:

1. **центры базисных функций** — производная выражается тем же интегралом со
   сдвинутым угловым моментом (``2α·I(l+1) − l·I(l−1)``);
2. **положения ядер в операторе притяжения** — отдельное слагаемое
   (Helgaker 9.9.21); без него градиент неверен даже для H₂⁺;
3. **межъядерное отталкивание**.

Стоимость: три сборки производного тензора ERI (по одной на ось), независимо от
числа атомов — производные по остальным трём центрам квартета получаются
перестановкой индексов благодаря 8-кратной симметрии. Соотношения
перестановки не принимаются на веру: итоговый градиент сверяется с конечными
разностями в ``tests/test_gradients.py``.

Единицы: энергия — хартри, градиент — хартри/бор (контракт ``GradientEngine``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quantumlab.domain.molecule import Molecule
from quantumlab.engine import integrals
from quantumlab.engine.basis import BasisSet
from quantumlab.engine.constants import angstrom_to_bohr
from quantumlab.engine.scf import ScfResult

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RhfGradient:
    """Энергия и градиент по декартовым координатам ядер."""

    energy_hartree: float
    gradient: Array

    @property
    def forces(self) -> Array:
        """Силы: ``F = −dE/dR``."""
        return np.asarray(-self.gradient)

    @property
    def max_force(self) -> float:
        """Наибольшая по модулю компонента силы — главный критерий сходимости."""
        return float(np.max(np.abs(self.gradient)))

    @property
    def rms_force(self) -> float:
        """Среднеквадратичная сила по всем компонентам."""
        return float(np.sqrt(np.mean(np.square(self.gradient))))


def energy_weighted_density(scf: ScfResult, n_occupied: int) -> Array:
    """Энерговзвешенная плотность ``W_μν = 2Σ_i^{занято} C_μi ε_i C_νi``."""
    occupied = scf.coefficients[:, :n_occupied]
    energies = np.asarray(scf.orbital_energies[:n_occupied])
    return np.asarray(2.0 * (occupied * energies) @ occupied.T)


def nuclear_repulsion_gradient(molecule: Molecule) -> Array:
    """Градиент межъядерного отталкивания, хартри/бор.

    ``E_nuc = Σ_{A<B} Z_A Z_B / R_AB`` ⇒ ``dE/dA_x = −Σ_{B≠A} Z_A Z_B (A_x−B_x)/R³``.
    """
    positions = [
        np.array([angstrom_to_bohr(value) for value in atom.position]) for atom in molecule.atoms
    ]
    charges = [atom.z for atom in molecule.atoms]
    gradient = np.zeros((len(positions), 3))
    for a, position_a in enumerate(positions):
        for b, position_b in enumerate(positions):
            if a == b:
                continue
            delta = position_a - position_b
            distance = float(np.linalg.norm(delta))
            gradient[a] -= charges[a] * charges[b] * delta / distance**3
    return gradient


def _function_owner(basis: BasisSet) -> Array:
    """Индекс атома-владельца для каждой базисной функции."""
    return np.array(
        [shell.center for shell in basis.shells for _ in range(shell.n_cartesian)], dtype=int
    )


def rhf_gradient(basis: BasisSet, molecule: Molecule, scf: ScfResult) -> RhfGradient:
    """Аналитический градиент RHF-энергии по координатам всех ядер.

    Требует **сошедшейся** SCF: формула использует стационарность энергии по
    орбиталям, поэтому на несошедшейся плотности градиент неверен. Проверка
    вызывающим кодом, а не здесь — движок решает, что делать с несошедшимся
    расчётом (прервать или сообщить).
    """
    n_occupied = molecule.n_electrons // 2
    density = scf.density
    weight = energy_weighted_density(scf, n_occupied)
    owner = _function_owner(basis)
    n_functions = basis.n_functions

    gradient = nuclear_repulsion_gradient(molecule)
    masks = [owner == atom for atom in range(len(molecule.atoms))]

    for axis in range(3):
        # Одна сборка производных на ось: производная по центру бра. Остальные
        # центры получаются транспозицией — это экономит вычисление интегралов.
        kinetic = integrals.build_kinetic_derivative(basis, molecule, axis)
        attraction = integrals.build_nuclear_attraction_center_derivative(basis, molecule, axis)
        core_bra = kinetic + attraction
        overlap_bra = integrals.build_overlap_derivative(basis, molecule, axis)
        eri_bra = integrals.build_electron_repulsion_derivative(basis, molecule, axis)

        # (μν|λσ) → производные по центру ν, λ, σ. Соотношения перестановки
        # проверяются тестом через сверку с конечными разностями.
        eri_slots = (
            eri_bra,
            eri_bra.transpose(1, 0, 2, 3),
            eri_bra.transpose(2, 3, 0, 1),
            eri_bra.transpose(2, 3, 1, 0),
        )

        for atom_index, mask in enumerate(masks):
            bra = mask[:, None]
            ket = mask[None, :]

            core = bra * core_bra + ket * core_bra.T
            overlap = bra * overlap_bra + ket * overlap_bra.T

            derivative = np.zeros((n_functions,) * 4)
            for slot, tensor in enumerate(eri_slots):
                shape = [1, 1, 1, 1]
                shape[slot] = n_functions
                derivative += mask.reshape(shape) * tensor

            one_electron = float(np.sum(density * core))
            coulomb = float(np.einsum("uv,ls,uvls", density, density, derivative))
            exchange = float(np.einsum("uv,ls,ulvs", density, density, derivative, optimize=True))
            orbital_relaxation = float(np.sum(weight * overlap))

            gradient[atom_index, axis] += (
                one_electron + 0.5 * coulomb - 0.25 * exchange - orbital_relaxation
            )

        # Движение самих ядер в операторе притяжения — отдельно для каждого атома:
        # это слагаемое не сводится к производной по центру базисной функции.
        for atom_index in range(len(molecule.atoms)):
            position_derivative = integrals.build_nuclear_attraction_position_derivative(
                basis, molecule, atom_index, axis
            )
            gradient[atom_index, axis] += float(np.sum(density * position_derivative))

    return RhfGradient(energy_hartree=scf.total_energy, gradient=gradient)
