"""Сферическая угловая схема: матрицы перехода и согласованность с декартовой.

Независимый оракул — PySCF (``cart=False``); без него проверяются алгебраические
тождества, которые не нуждаются во внешнем пакете.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.engine import integrals
from quantumlab.engine.basis import (
    build_basis,
    cartesian_powers,
    spherical_transformation,
)
from quantumlab.engine.functional import evaluate_basis_with_gradients
from quantumlab.engine.scf import ScfSettings, run_rhf

FIXTURES = Path(__file__).parent / "fixtures"
TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


@pytest.mark.parametrize("l_value", range(0, 5))
def test_transformation_has_the_right_shape_and_orthonormal_columns(l_value: int) -> None:
    """``T`` — ``(n_cart, 2l+1)``, а ``Tᵀ S_shell T = 1`` для перекрывания оболочки."""
    transform = spherical_transformation(l_value)
    assert transform.shape == ((l_value + 1) * (l_value + 2) // 2, 2 * l_value + 1)
    # Перекрывание нормированных декартовых функций одной оболочки в её центре.
    shell_basis = build_basis("cc-pvqz", Molecule.from_atoms(["C"], [(0.0, 0.0, 0.0)]))
    shells = [s for s in shell_basis.shells if s.angular_momentum == l_value]
    assert shells, "в cc-pVQZ/C должна быть оболочка такого момента"
    gram = integrals.build_overlap(
        type(shell_basis)(
            name="t",
            display_name="t",
            shells=(shells[0],),
        ),
        Molecule.from_atoms(["C"], [(0.0, 0.0, 0.0)]),
    )
    assert gram.shape[0] == len(cartesian_powers(l_value))
    assert np.allclose(transform.T @ gram @ transform, np.eye(2 * l_value + 1), atol=1e-12)


def test_functions_are_harmonic_polynomials() -> None:
    """Оболочки ``d`` и выше отдают именно сферические гармоники: лапласиан угловой части = 0.

    ``Δ(x^a y^b z^c)`` складывается из членов со степенями на 2 ниже; для
    гармонического многочлена все коэффициенты при сумме обязаны обнулиться.
    """
    for l_value in (2, 3, 4):
        powers = cartesian_powers(l_value)
        transform = spherical_transformation(l_value)
        from quantumlab.engine.basis import _component_scale

        scales = np.array([_component_scale(l_value, p) for p in powers])
        polynomial = transform * scales[:, None]  # коэффициенты при x^a y^b z^c
        reduced = cartesian_powers(l_value - 2)
        index = {p: i for i, p in enumerate(reduced)}
        laplacian = np.zeros((len(reduced), polynomial.shape[1]))
        for i, (a, b, c) in enumerate(powers):
            for axis, power in enumerate((a, b, c)):
                if power >= 2:
                    target = [a, b, c]
                    target[axis] -= 2
                    laplacian[index[(target[0], target[1], target[2])]] += (
                        power * (power - 1) * polynomial[i]
                    )
        assert np.allclose(laplacian, 0.0, atol=1e-12), l_value


def test_basis_sizes_follow_the_scheme() -> None:
    """Вода/cc-pVDZ: 24 сферических функции против 25 декартовых; sto-3g — без изменений."""
    water = _water()
    spherical = build_basis("cc-pvdz", water)
    assert spherical.spherical
    assert spherical.n_functions == 24
    assert spherical.cartesian().n_functions == 25
    assert build_basis("cc-pvdz", water, spherical=False).n_functions == 25
    assert not build_basis("sto-3g", water).spherical


def test_spherical_overlap_is_unit_on_the_diagonal() -> None:
    """Каждая сферическая функция нормирована: ``diag(S) = 1``."""
    water = _water()
    overlap = integrals.build_overlap(build_basis("def2-svp", water), water)
    assert np.allclose(np.diag(overlap), 1.0, atol=1e-10)


def test_grid_values_match_the_integral_transform() -> None:
    """Значения AO на сетке и интегралы используют одну матрицу перехода."""
    water = _water()
    spherical = build_basis("cc-pvdz", water)
    points = np.random.default_rng(3).normal(size=(40, 3))
    values, gradients = evaluate_basis_with_gradients(spherical, water, points)
    cart_values, cart_gradients = evaluate_basis_with_gradients(
        spherical.cartesian(), water, points
    )
    transform = spherical.transformation_matrix()
    assert np.allclose(values, cart_values @ transform)
    assert np.allclose(gradients, np.einsum("pcb,cm->pmb", cart_gradients, transform))


def test_derivative_integrals_reject_spherical_basis() -> None:
    """Производные интегралов определены в декартовой схеме; молчаливой ошибки быть не должно."""
    water = _water()
    with pytest.raises(ValueError, match="декартовой"):
        integrals.build_overlap_derivative(build_basis("cc-pvdz", water), water, 0)


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [("cc-pvdz", "cc-pVDZ"), ("def2-svp", "def2-SVP"), ("6-311g(d,p)", "6-311G**")],
)
def test_spherical_energy_matches_pyscf(ours: str, theirs: str) -> None:
    """RHF-энергия в сферической схеме совпадает с PySCF (``cart=False``)."""
    pyscf = pytest.importorskip("pyscf")
    importlib.import_module("pyscf.gto")
    molecule = _water()
    atom = "; ".join(
        f"{a.symbol} {a.position[0]} {a.position[1]} {a.position[2]}" for a in molecule.atoms
    )
    reference = pyscf.gto.M(atom=atom, basis=theirs, cart=False, unit="Angstrom", verbose=0)
    expected = reference.RHF().run(conv_tol=1e-12, max_cycle=300).e_tot
    result = run_rhf(build_basis(ours, molecule), molecule, TIGHT)
    assert result.converged
    assert result.total_energy == pytest.approx(expected, abs=1e-7)


def _pyscf_water(basis: str, *, spin: int = 0, charge: int = 0) -> object:
    pyscf = pytest.importorskip("pyscf")
    importlib.import_module("pyscf.gto")
    molecule = _water()
    atom = "; ".join(
        f"{a.symbol} {a.position[0]} {a.position[1]} {a.position[2]}" for a in molecule.atoms
    )
    return pyscf.gto.M(
        atom=atom, basis=basis, cart=False, unit="Angstrom", spin=spin, charge=charge, verbose=0
    )


@pytest.mark.scientific
def test_spherical_rhf_gradient_matches_pyscf() -> None:
    """Аналитический градиент RHF в сферической схеме (плотность переводится в декартову)."""
    from quantumlab.engine.gradients import rhf_gradient

    molecule = _water()
    basis = build_basis("cc-pvdz", molecule)
    result = run_rhf(basis, molecule, TIGHT)
    ours = rhf_gradient(basis, molecule, result).gradient
    reference = _pyscf_water("cc-pVDZ").RHF().run(conv_tol=1e-12)  # type: ignore[attr-defined]
    theirs = np.asarray(reference.nuc_grad_method().kernel())
    assert np.max(np.abs(ours - theirs)) < 5e-7


@pytest.mark.scientific
def test_spherical_rks_gradient_matches_pyscf() -> None:
    """Градиент GGA/PBE в сферической схеме: XC-вклад использует декартовы гессианы AO."""
    from quantumlab.domain.spec import GridPreset
    from quantumlab.engine.dft import run_rks
    from quantumlab.engine.functional import get_functional
    from quantumlab.engine.gradients import rks_gradient
    from quantumlab.engine.quadrature import build_grid

    pyscf_dft = importlib.import_module("pyscf.dft")
    molecule = _water()
    basis = build_basis("def2-svp", molecule)
    functional = get_functional("pbe")
    grid = build_grid(molecule, GridPreset.ULTRAFINE)
    result = run_rks(basis, molecule, functional, grid=grid)
    ours = rks_gradient(basis, molecule, result, grid, functional).gradient
    scf = pyscf_dft.RKS(_pyscf_water("def2-SVP"))
    scf.xc = "PBE"
    scf.grids.atom_grid = (120, 974)
    scf.conv_tol = 1e-12
    scf.run()
    gradient = scf.nuc_grad_method()
    gradient.grid_response = False
    theirs = np.asarray(gradient.kernel())
    assert np.max(np.abs(ours - theirs)) < 5e-6
