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
    run_rhf,
    run_rohf,
    run_uhf,
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


def test_energy_decomposition_identity(rhf_sto3g: ScfResult, water: Molecule) -> None:
    """E = T + V_яд-эл + V_эл-эл + V_яд-яд, пересчитанное по плотности.

    Точное тождество: оно обязано выполняться с машинной точностью и ловит
    ошибки в любом из слагаемых (нормировка, множитель ½ у обменного вклада,
    потеря ядерного отталкивания).
    """
    basis = build_basis("sto-3g", water)
    density = rhf_sto3g.density
    kinetic = float(np.sum(density * build_kinetic(basis, water)))
    from quantumlab.engine.integrals import build_electron_repulsion, build_nuclear_attraction

    attraction = float(np.sum(density * build_nuclear_attraction(basis, water)))
    repulsion = build_electron_repulsion(basis, water)
    coulomb = float(np.einsum("uv,ls,uvls", density, density, repulsion))
    exchange = float(np.einsum("uv,ls,ulvs", density, density, repulsion))
    electron_electron = 0.5 * (coulomb - 0.5 * exchange)

    total = kinetic + attraction + electron_electron + nuclear_repulsion(water)
    assert total == pytest.approx(rhf_sto3g.total_energy, abs=1e-10)


def test_virial_ratio_is_not_a_correctness_criterion(water: Molecule) -> None:
    """Отношение −V/T **не равно** 2 в конечном базисе — и это не ошибка.

    Теорема вириала 2⟨T⟩ + ⟨V⟩ = −Σ_A R_A·∂E/∂R_A выводится масштабированием
    всех координат, то есть предполагает, что базис умеет масштабироваться
    вместе с геометрией. При фиксированном конечном базисе волновая функция
    масштабироваться не может, и отклонение отражает негибкость базиса.

    Численно: H₂/STO-3G в собственном минимуме даёт −V/T = 1.924772, и PySCF
    как независимый оракул даёт ровно то же значение (T = 1.208412113,
    E = −1.117505885). Поэтому проверка качества с фиксированным порогом
    вокруг 2 была бы ложным контролем: она «проходила» на воде случайно и
    поднимала бы тревогу на корректном расчёте H₂. Такой проверки в движке
    нет намеренно — вместо неё используются точные тождества.
    """
    hydrogen = Molecule(
        name="h2",
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, 0.712230)),
        ),
    )
    basis = build_basis("sto-3g", hydrogen)
    scf = run_rhf(basis, hydrogen, ScfSettings())
    kinetic = float(np.sum(scf.density * build_kinetic(basis, hydrogen)))
    potential = scf.total_energy - kinetic
    # Значение сверено с PySCF; фиксированный порог «2.0 ± 0.02» его отверг бы.
    assert -potential / kinetic == pytest.approx(1.924772, abs=1e-5)

    # Для воды то же отношение случайно оказывается близко к 2 — именно поэтому
    # такая проверка выглядит правдоподобно, хотя критерием не является.
    water_basis = build_basis("sto-3g", water)
    water_scf = run_rhf(basis=water_basis, molecule=water, settings=ScfSettings())
    water_kinetic = float(np.sum(water_scf.density * build_kinetic(water_basis, water)))
    water_potential = water_scf.total_energy - water_kinetic
    assert -water_potential / water_kinetic == pytest.approx(2.0, abs=0.02)


def test_rohf_closed_shell_reduces_exactly_to_rhf(water: Molecule) -> None:
    """ROHF замкнутой оболочки совпадает с RHF.

    Не приближённо, а точно: при ``n_alpha = n_beta`` открытый проектор
    зануляется, ``Fc = Fα = Fβ``, и поскольку ``Pc + Pv = I``, эффективный
    фокиан сворачивается в ``Fc``. Расхождение означало бы ошибку в
    проекторах, а не численный шум.
    """
    basis = build_basis("sto-3g", water)
    rohf = run_rohf(basis, water, TIGHT)
    rhf = run_rhf(basis, water, TIGHT)
    assert rohf.converged
    assert abs(rohf.total_energy - rhf.total_energy) < 1e-11


def test_rohf_is_a_spin_eigenfunction() -> None:
    """⟨S²⟩ = S(S+1) точно — определяющее свойство ROHF.

    В UHF та же система даёт спиновое загрязнение (⟨S²⟩ выше S(S+1)). Ровное
    равенство — следствие общих орбиталей: замкнутая часть совпадает в обоих
    каналах, и сумма квадратов перекрытий сокращает ``n_beta``. Проверка
    независима от эталонной энергии и ловит подмену ROHF на UHF.
    """
    molecule = Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), multiplicity=2
    )
    basis = build_basis("sto-3g", molecule)
    rohf = run_rohf(basis, molecule, TIGHT)
    exact = 0.5 * 1.5
    assert rohf.converged
    assert rohf.s_squared == pytest.approx(exact, abs=1e-10)

    uhf = run_uhf(basis, molecule, TIGHT)
    assert uhf.s_squared > exact + 1e-4, "UHF обязан давать загрязнение"


def test_rohf_energy_is_above_uhf_and_matches_pyscf() -> None:
    """Энергия ROHF выше UHF (метод более ограничен) и совпадает с оракулом.

    Вариационный принцип: ROHF допускает меньшее множество плотностей, чем UHF,
    поэтому его энергия не может быть ниже. Совпадение с независимой реализацией
    подтверждает, что выше не потому, что расчёт неверен.
    """
    pyscf = pytest.importorskip("pyscf")
    molecule = Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), multiplicity=2
    )
    basis = build_basis("sto-3g", molecule)
    ours = run_rohf(basis, molecule, TIGHT).total_energy
    uhf_energy = run_uhf(basis, molecule, TIGHT).total_energy
    assert ours > uhf_energy

    atom_string = "; ".join(
        f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}"
        for atom in molecule.atoms
    )
    reference = pyscf.scf.ROHF(
        pyscf.gto.M(atom=atom_string, basis="sto-3g", cart=True, spin=1, verbose=0)
    ).run(conv_tol=1e-12)
    assert ours == pytest.approx(float(reference.e_tot), abs=1e-7)
