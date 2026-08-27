"""Тесты RHF: физические инварианты и поведение итерационного процесса."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.engine.basis import build_basis, nuclear_repulsion
from quantumlab.engine.integrals import build_kinetic, build_overlap
from quantumlab.engine.scf import (
    ScfResult,
    ScfSettings,
    canonical_orthogonalizer,
    density_from_coefficients,
    run_rhf,
)

FIXTURES = Path(__file__).parent / "fixtures"
TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)


@pytest.fixture(scope="module")
def water() -> Molecule:
    """Молекула воды из фикстуры."""
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


@pytest.fixture(scope="module")
def rhf_sto3g(water: Molecule) -> ScfResult:
    """Сходившийся RHF/STO-3G для воды."""
    return run_rhf(build_basis("sto-3g", water), water, TIGHT)


def test_scf_converges(rhf_sto3g: ScfResult) -> None:
    """SCF обязан сойтись на воде/STO-3G."""
    assert rhf_sto3g.converged is True
    assert rhf_sto3g.iterations < 50


def test_total_energy_is_electronic_plus_nuclear(rhf_sto3g: ScfResult, water: Molecule) -> None:
    """E = E_эл + E_яд — тривиально, но ловит потерю слагаемого."""
    assert rhf_sto3g.total_energy == pytest.approx(
        rhf_sto3g.electronic_energy + nuclear_repulsion(water), abs=1e-12
    )


def test_density_gives_correct_electron_count(rhf_sto3g: ScfResult, water: Molecule) -> None:
    """tr(D S) = N_e: плотность описывает ровно все электроны."""
    basis = build_basis("sto-3g", water)
    overlap = build_overlap(basis, water)
    assert float(np.trace(rhf_sto3g.density @ overlap)) == pytest.approx(
        water.n_electrons, abs=1e-8
    )


def test_density_is_idempotent_in_orthogonal_basis(rhf_sto3g: ScfResult, water: Molecule) -> None:
    """В ортогональном базисе D'² = 2 D' (двойное занятие)."""
    basis = build_basis("sto-3g", water)
    overlap = build_overlap(basis, water)
    x = canonical_orthogonalizer(overlap)
    x_inverse = np.linalg.inv(x)
    density_prime = x_inverse @ rhf_sto3g.density @ x_inverse.T
    assert np.allclose(density_prime @ density_prime, 2.0 * density_prime, atol=1e-8)


def test_coefficients_are_orthonormal(rhf_sto3g: ScfResult, water: Molecule) -> None:
    """C^T S C = I."""
    basis = build_basis("sto-3g", water)
    overlap = build_overlap(basis, water)
    product = rhf_sto3g.coefficients.T @ overlap @ rhf_sto3g.coefficients
    assert np.allclose(product, np.eye(product.shape[0]), atol=1e-8)


def test_virial_theorem(rhf_sto3g: ScfResult, water: Molecule) -> None:
    """Теорема вириала: для равновесной геометрии E = −T, то есть 2T + V = 0.

    Проверяется на сходятемся решении: T — кинетическая энергия, V — сумма
    электрон-ядерного, электрон-электронного и ядерного вкладов. Отклонение от
    нуля — прямой индикатор ошибки в одноэлектронных интегралах.
    """
    basis = build_basis("sto-3g", water)
    kinetic = build_kinetic(basis, water)
    density = rhf_sto3g.density
    t = float(np.sum(density * kinetic))
    total_potential = rhf_sto3g.total_energy - t
    # Теорема вириала: 2⟨T⟩ + ⟨V⟩ = −Σ_A R_A · ∂E/∂R_A. В равновесной геометрии
    # правая часть равна нулю и ⟨V⟩/⟨T⟩ = −2. Геометрия из фикстуры не
    # оптимизирована, поэтому допускается небольшое отклонение; сам факт, что
    # отношение близко к −2, ловит ошибки в одноэлектронных интегралах.
    ratio = -total_potential / t
    assert ratio == pytest.approx(2.0, abs=0.02)


def test_orbital_energies_are_sorted(rhf_sto3g: ScfResult) -> None:
    """Орбитальные энергии идут по возрастанию."""
    energies = list(rhf_sto3g.orbital_energies)
    assert energies == sorted(energies)


def test_homo_lumo_gap_is_reported(rhf_sto3g: ScfResult) -> None:
    """Вода/STO-3G: 5 занятых орбиталей, есть разрыв ГЗМО-НСМО."""
    occupied = rhf_sto3g.orbital_energies[:5]
    virtual = rhf_sto3g.orbital_energies[5:]
    assert occupied[-1] < 0 < virtual[0]


def test_energy_decreases_and_is_stable(water: Molecule) -> None:
    """Энергия STO-3G → 6-31G → 6-31G(d,p) монотонно снижается (вариационный принцип)."""
    energies = [
        run_rhf(build_basis(name, water), water, TIGHT).total_energy
        for name in ("sto-3g", "6-31g", "6-31g(d,p)")
    ]
    assert energies[0] > energies[1] > energies[2]


def test_strategies_are_recorded(rhf_sto3g: ScfResult) -> None:
    """Протокол стратегий честный: стартовое приближение и реально применённый DIIS."""
    assert "core-hamiltonian-guess" in rhf_sto3g.strategies_used
    assert "diis" in rhf_sto3g.strategies_used


def test_history_is_complete(rhf_sto3g: ScfResult) -> None:
    """История содержит столько же записей, сколько итераций."""
    assert len(rhf_sto3g.history) == rhf_sto3g.iterations
    assert rhf_sto3g.history[0].iteration == 1


def test_odd_electron_count_is_rejected() -> None:
    """Нечётное число электронов — явная ошибка, а не молчаливый неверный результат."""
    hydrogen = Molecule(
        name="h", atoms=(Atom(symbol="H", position=(0.0, 0.0, 0.0)),), multiplicity=2
    )
    basis = build_basis("sto-3g", hydrogen)
    with pytest.raises(ValueError, match="чётного числа электронов"):
        run_rhf(basis, hydrogen)


def test_canonical_orthogonalizer_inverts_overlap(water: Molecule) -> None:
    """X^T S X = I."""
    basis = build_basis("sto-3g", water)
    overlap = build_overlap(basis, water)
    x = canonical_orthogonalizer(overlap)
    assert np.allclose(x.T @ overlap @ x, np.eye(overlap.shape[0]), atol=1e-10)


def test_density_from_coefficients_doubles_occupation() -> None:
    """D = 2 C_occ C_occ^T."""
    coefficients = np.eye(4)
    density = density_from_coefficients(coefficients, 2)
    assert np.allclose(density, np.diag([2.0, 2.0, 0.0, 0.0]))
