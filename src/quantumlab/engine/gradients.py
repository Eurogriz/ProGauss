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
from typing import Protocol

import numpy as np
import numpy.typing as npt

from quantumlab.domain.molecule import Molecule
from quantumlab.engine import integrals
from quantumlab.engine.basis import BasisSet
from quantumlab.engine.constants import angstrom_to_bohr
from quantumlab.engine.contracts import ExchangeCorrelationFunctional
from quantumlab.engine.dft import RksResult
from quantumlab.engine.functional import (
    density_at_points,
    density_gradient_at_points,
    evaluate_basis_hessian_for_center,
    evaluate_basis_with_gradients,
)
from quantumlab.engine.quadrature import QuadratureGrid
from quantumlab.engine.scf import ScfResult, UhfResult, spin_population

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


class OrbitalSolution(Protocol):
    """Всё, что нужно от решателя для энерговзвешенной плотности.

    Структурный тип, а не ``ScfResult``: ``RksResult`` повторяет набор полей
    RHF, но не наследует от него, и привязка к конкретному классу заставила бы
    DFT-градиент дублировать вычисление ``W``.
    """

    coefficients: np.ndarray
    orbital_energies: tuple[float, ...]
    density: np.ndarray


def energy_weighted_density(solution: OrbitalSolution, n_occupied: int) -> Array:
    """Энерговзвешенная плотность ``W_μν = 2Σ_i^{занято} C_μi ε_i C_νi``."""
    occupied = solution.coefficients[:, :n_occupied]
    energies = np.asarray(solution.orbital_energies[:n_occupied])
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


def _orbital_gradient(
    basis: BasisSet,
    molecule: Molecule,
    *,
    density: Array,
    exchange_densities: tuple[Array, ...],
    weight: Array,
    exchange_coefficient: float,
) -> Array:
    """Аналитический градиент электронной части, хартри/бор.

    Требует **сошедшегося** решения: формула использует стационарность энергии по
    орбиталям, поэтому на несошедшейся плотности градиент неверен. Проверка
    вызывающим кодом, а не здесь — движок решает, что делать с несошедшимся
    расчётом (прервать или сообщить).

    Параметры разделены по смыслу, а не по методу, чтобы один цикл служил RHF,
    RKS и UHF:

    * ``density`` — та, что входит в одноэлектронную часть и в кулоновский член.
      У UHF это сумма каналов ``D^α + D^β``.
    * ``exchange_densities`` — плотности, по которым строится обмен. У RHF и RKS
      одна (полная), у UHF две (по каналу на каждый): электрон обменивается
      только с электронами своего спина.
    * ``exchange_coefficient`` — множитель при ``Σ DD·(μλ|νσ)'``. У RHF и UHF
      это ``¼``; у гибридного функционала обмен точный лишь частично, и
      множитель равен ``α/4``. Чистые функционалы DFT передают ``0``: у них
      обмен целиком содержится в ``E_xc`` и входит через отдельное слагаемое.
    * ``weight`` — энерговзвешенная плотность, входящая в член релаксации
      орбиталей ``Σ W·S'``.
    """
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
            exchange = 0.0
            for exchange_density in exchange_densities:
                exchange += float(
                    np.einsum(
                        "uv,ls,ulvs", exchange_density, exchange_density, derivative, optimize=True
                    )
                )
            orbital_relaxation = float(np.sum(weight * overlap))

            gradient[atom_index, axis] += (
                one_electron + 0.5 * coulomb - exchange_coefficient * exchange - orbital_relaxation
            )

        # Движение самих ядер в операторе притяжения — отдельно для каждого атома:
        # это слагаемое не сводится к производной по центру базисной функции.
        for atom_index in range(len(molecule.atoms)):
            position_derivative = integrals.build_nuclear_attraction_position_derivative(
                basis, molecule, atom_index, axis
            )
            gradient[atom_index, axis] += float(np.sum(density * position_derivative))

    return gradient


def rhf_gradient(basis: BasisSet, molecule: Molecule, scf: ScfResult) -> RhfGradient:
    """Аналитический градиент RHF-энергии по координатам всех ядер."""
    n_occupied = molecule.n_electrons // 2
    gradient = _orbital_gradient(
        basis,
        molecule,
        density=scf.density,
        exchange_densities=(scf.density,),
        weight=energy_weighted_density(scf, n_occupied),
        exchange_coefficient=0.25,
    )
    return RhfGradient(energy_hartree=scf.total_energy, gradient=gradient)


def uhf_gradient(basis: BasisSet, molecule: Molecule, uhf: UhfResult) -> RhfGradient:
    """Аналитический градиент UHF-энергии по координатам всех ядер.

    Одноэлектронная и кулоновская части строятся на полной плотности
    ``D = D^α + D^β``, а обменная — на односпиновых:

    ``E = Σ D·H + ½ ΣΣ DD·(μν|λσ) − ½ ΣΣ D^αD^α·(μλ|νσ) − ½ ΣΣ D^βD^β·(μλ|νσ)``

    Складывать каналы в одну плотность **до** обмена нельзя: это добавило бы
    межспиновое обменное взаимодействие ``−½ ΣΣ D^αD^β·(μλ|νσ)``, которого в
    UHF нет.

    Коэффициент при обмене — **½ на канал**, а не ¼ как в RHF. Это следует из
    фокиана ``F^σ = H + J − K(D^σ)`` (в RHF — ``H + J − ½K``): спин-орбиталь
    занята один раз, а не двумя электронами. Проверка — закрытая оболочка, где
    ``D^α = D^β = D/2`` и оба выражения обязаны совпасть:
    ``−½·2·(D/2)K(D/2) = −¼ D·K(D)``.

    Член релаксации орбиталей берётся с множителем 1 на спин-орбиталь по той же
    причине.

    Требует сошедшегося расчёта по той же причине, что и ``rhf_gradient``.
    """
    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)
    alpha_occupied = uhf.alpha_coefficients[:, :n_alpha]
    beta_occupied = uhf.beta_coefficients[:, :n_beta]
    weight = np.asarray(
        (alpha_occupied * np.asarray(uhf.alpha_energies[:n_alpha])) @ alpha_occupied.T
        + (beta_occupied * np.asarray(uhf.beta_energies[:n_beta])) @ beta_occupied.T
    )
    gradient = _orbital_gradient(
        basis,
        molecule,
        density=uhf.density_alpha + uhf.density_beta,
        exchange_densities=(uhf.density_alpha, uhf.density_beta),
        weight=weight,
        exchange_coefficient=0.5,
    )
    return RhfGradient(energy_hartree=uhf.total_energy, gradient=gradient)


def xc_gradient(
    basis: BasisSet,
    molecule: Molecule,
    grid: QuadratureGrid,
    density: Array,
    functional: ExchangeCorrelationFunctional,
) -> Array:
    """Обменно-корреляционный вклад в градиент, хартри/бор.

    ``E_xc = Σ_g w_g ρ_g ε_xc(ρ_g, σ_g)`` ⇒

    ``dE_xc/dR_Aa = Σ_g w_g [ v_ρ ∂ρ_g/∂R_Aa + v_σ ∂σ_g/∂R_Aa ]``

    Производные плотности — от движения базисных функций, центрированных на
    атоме ``A``, при **неподвижных точках сетки**:

    ``∂ρ/∂R_Aa = −2 Σ_{μ∈A} Σ_ν D_μν (∂_aφ_μ) φ_ν``

    Знак минус потому, что ``φ_μ(r − R_A)`` сдвигается вместе с центром. Для
    GGA добавляется член с ``∂σ/∂R_Aa = 2∇ρ·∂(∇ρ)/∂R_Aa``, куда входят вторые
    производные базисных функций.

    **Градиент соответствует неподвижной в пространстве сетке.** Точки Беке
    центрированы на атомах, поэтому при движении ядра сетка, построенная заново,
    смещается вместе с ним: ``dE_xc/dR`` для такой поверхности содержало бы ещё
    отклик точек и весов, ``Σ_g (∂(w_g)/∂R_Aa) ρ_g ε_g`` плюс сдвиг самих ``r_g``.
    Движок перестраивает сетку на каждой геометрии (иначе энергия перестала бы
    быть той величиной, что в расчёте в одной точке), поэтому между поверхностью,
    для которой градиент точен, и поверхностью оптимизатора есть расхождение.

    Оно измерено, а не оценено. На воде/STO-3G/SVWN полное расхождение
    (перестраиваемая сетка против замороженной) — ``max|Δg| = 7.0e−06``
    хартри/бор, то есть в 64 раза ниже порога ``max_force = 4.5e−4``. Основная
    часть — сдвиг точек, а не производные весов Беке: у PySCF на сопоставимых
    сетках член ``grid_response`` даёт лишь ``2.5e−08 … 3.8e−08``. Приближение
    стандартное: в Molpro производные весов сетки в аналитическом градиенте
    вообще выключены по умолчанию (``GRIDGRAD=0``).
    """
    values, basis_gradients = evaluate_basis_with_gradients(basis, molecule, grid.points)
    rho = density_at_points(values, density)
    rho_gradient = density_gradient_at_points(values, basis_gradients, density)
    evaluation = functional.evaluate(grid.points, rho, rho_gradient)

    v_rho = np.asarray(evaluation.vrho)
    v_sigma = np.asarray(evaluation.vsigma) if evaluation.vsigma is not None else None
    weighted_v_rho = grid.weights * v_rho

    contracted = np.asarray(density @ values.T)
    gradient_contracted = (
        np.einsum("nm,pnb->mpb", density, basis_gradients) if v_sigma is not None else None
    )

    owner = _function_owner(basis)
    gradient = np.zeros((len(molecule.atoms), 3))
    for atom in range(len(molecule.atoms)):
        columns = np.flatnonzero(owner == atom)
        if columns.size == 0:
            continue
        hessian = (
            evaluate_basis_hessian_for_center(basis, molecule, grid.points, atom)
            if v_sigma is not None
            else None
        )
        for axis in range(3):
            d_phi = basis_gradients[:, columns, axis]
            term = -2.0 * float(
                np.sum(weighted_v_rho * np.sum(d_phi * contracted[columns].T, axis=1))
            )
            if v_sigma is not None and hessian is not None and gradient_contracted is not None:
                # Ось ``axis`` срезается до einsum явно: индекс, отсутствующий
                # в выходной части подписи, einsum считает суммируемым, и
                # "pjab,jp->pb" сложило бы производные по всем трём осям.
                first = np.einsum("pjb,jp->pb", hessian[:, :, axis, :], contracted[columns])
                second = np.einsum("pj,jpb->pb", d_phi, gradient_contracted[columns])
                inner = np.sum(rho_gradient * (first + second), axis=1)
                term += -4.0 * float(np.sum(grid.weights * v_sigma * inner))
            gradient[atom, axis] = term
    return gradient


def rks_gradient(
    basis: BasisSet,
    molecule: Molecule,
    rks: RksResult,
    grid: QuadratureGrid,
    functional: ExchangeCorrelationFunctional,
) -> RhfGradient:
    """Аналитический градиент RKS-энергии по координатам всех ядер.

    Орбитальная часть совпадает с RHF с точностью до множителя точного обмена:
    у гибрида обменный интеграл входит с весом ``α/4`` вместо ``¼``, а у чистого
    функционала — не входит вовсе. К ней добавляется обменно-корреляционный
    вклад, вычисленный на той же сетке, что и сама энергия.
    """
    alpha = float(rks.exact_exchange_fraction)
    n_occupied = molecule.n_electrons // 2
    gradient = _orbital_gradient(
        basis,
        molecule,
        density=rks.density,
        exchange_densities=(rks.density,),
        weight=energy_weighted_density(rks, n_occupied),
        exchange_coefficient=0.25 * alpha,
    )
    gradient = gradient + xc_gradient(basis, molecule, grid, rks.density, functional)
    return RhfGradient(energy_hartree=rks.total_energy, gradient=gradient)
