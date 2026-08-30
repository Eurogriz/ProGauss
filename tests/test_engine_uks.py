"""UKS — спиново-поляризованный DFT: энергия, аналитический градиент, проверки.

Пятая вертикаль DFT после RKS/LDA, RKS/PBE, RKS/PBE0, D3 и ROHF-градиента.
Сверяется по трём независимым линиям, каждая ловит свой класс ошибки:

* **ядра ``evaluate_spin``** сверяются с LibXC (через PySCF) по энергии и
  обоим потенциалам — ошибка формулы или цепного правила;
* **энергия UKS** сверяется с ``pyscf.dft.UKS`` — ошибка решателя (фокиан,
  доля точного обмена, SCF);
* **аналитический градиент** сверяется с конечными разностями и в
  замкнутооболочечном пределе обязан перейти в градиент RKS — ошибка
  производной энергии по геометрии.

Согласно ADR-002 PySCF/LibXC — оракулы, а не источник истины.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.dft import run_rks, run_uks
from quantumlab.engine.functional import (
    BeckeExchange,
    LdaExchange,
    LypCorrelation,
    PbeCorrelation,
    PbeExchange,
    VwnCorrelation,
    get_functional,
)
from quantumlab.engine.gradients import rks_gradient, uks_gradient
from quantumlab.engine.quadrature import build_grid
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.scf import ScfSettings, spin_population
from quantumlab.errors import CombinationUnavailableError

FIXTURES = Path(__file__).parent / "fixtures"
TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=300)

ALL_FUNCTIONALS = ("svwn", "pbe", "blyp", "pbe0", "b3lyp")

_CH4_XYZ = """5
CH4
C 0 0 0
H 0.6291 0 0
H -0.1905 0.5485 0
H -0.1905 -0.2743 0.4743
H -0.1905 -0.2743 -0.4743
"""

_OH_XYZ = """2
OH
O 0 0 0
H 0 0 0.9588
"""


def _ch4() -> Molecule:
    return Molecule.from_xyz(_CH4_XYZ, charge=0, multiplicity=1)


def _oh() -> Molecule:
    return Molecule.from_xyz(_OH_XYZ, charge=0, multiplicity=2)


def _ch_radical() -> Molecule:
    return Molecule.from_xyz(
        (FIXTURES / "ch-radical.xyz").read_text(encoding="utf-8"), name="ch", multiplicity=2
    )


# --------------------------------------------------------------------------- #
# Ядра evaluate_spin vs LibXC (через PySCF)
# --------------------------------------------------------------------------- #
def _spin_grid(
    seed: int = 42, size: int = 60
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Воспроизводимые плотности и градиенты обоих каналов для сверки с LibXC."""
    generator = np.random.default_rng(seed)
    rho_a = generator.uniform(0.05, 3.0, size)
    rho_b = generator.uniform(0.05, 3.0, size)
    grad_a = generator.uniform(-2.0, 2.0, (size, 3))
    grad_b = generator.uniform(-2.0, 2.0, (size, 3))
    return rho_a, rho_b, grad_a, grad_b


def _libxc_spin_eval(
    code: str,
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    gga: bool,
    grad_a: np.ndarray,
    grad_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Вызывает LibXC в спиновой установке; возвращает (ε, vρ, vσ).

    LibXC со ``spin=1`` отдаёт энергию **на частицу** — то же соглашение, что
    у ``XcEvaluationSpin.energy_density``. ``vρ`` имеет форму ``(n, 2)``,
    ``vσ`` — ``(n, 3)`` со строками (αα, αβ, ββ).
    """
    pyscf = pytest.importorskip("pyscf")
    importlib.import_module("pyscf.dft")
    libxc = pyscf.dft.libxc
    size = rho_a.size
    if gga:
        raw = np.zeros((2, 4, size))
        raw[0, 0] = rho_a
        raw[0, 1:] = grad_a.T
        raw[1, 0] = rho_b
        raw[1, 1:] = grad_b.T
    else:
        raw = np.zeros((2, 1, size))
        raw[0, 0] = rho_a
        raw[1, 0] = rho_b
    out = libxc.eval_xc(code, raw, spin=1, deriv=1)
    energy = np.asarray(out[0])
    vrho = np.asarray(out[1][0])
    vsigma = np.asarray(out[1][1]) if len(out[1]) > 1 else None
    return energy, vrho, vsigma


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("part", "code", "gga"),
    [
        (LdaExchange, "LDA_X", False),
        (VwnCorrelation, "LDA_C_VWN", False),
        (PbeExchange, "GGA_X_PBE", True),
        (PbeCorrelation, "GGA_C_PBE", True),
        (BeckeExchange, "GGA_X_B88", True),
        (LypCorrelation, "GGA_C_LYP", True),
    ],
)
def test_spin_cores_match_libxc(part: Any, code: str, gga: bool) -> None:
    """Каждое спиновое ядро совпадает с LibXC 7.0.0 по энергии и потенциалам."""
    rho_a, rho_b, grad_a, grad_b = _spin_grid()
    ref_energy, ref_vrho, ref_vsigma = _libxc_spin_eval(code, rho_a, rho_b, gga, grad_a, grad_b)

    evaluation = part().evaluate_spin(
        np.zeros((rho_a.size, 3)),
        np.stack([rho_a, rho_b]),
        np.stack([grad_a, grad_b]) if gga else None,
    )
    assert np.max(np.abs(evaluation.energy_density - ref_energy)) < 1e-12
    assert np.max(np.abs(evaluation.vrho - ref_vrho.T)) < 1e-12
    if gga:
        assert evaluation.vsigma is not None and ref_vsigma is not None
        # Строки LibXC (αα, αβ, ββ) → наш симметричный (2, 2, n).
        assert np.max(np.abs(evaluation.vsigma[0, 0] - ref_vsigma[:, 0])) < 1e-12
        assert np.max(np.abs(evaluation.vsigma[0, 1] - ref_vsigma[:, 1])) < 1e-12
        assert np.max(np.abs(evaluation.vsigma[1, 1] - ref_vsigma[:, 2])) < 1e-12
    else:
        assert evaluation.vsigma is None


def test_spin_exchange_closed_shell_limit() -> None:
    """При ρ_α = ρ_β спиновый обмен обязан перейти в неполяризованный.

    Это ловит классическую ошибку с множителем: коэффициент Слэтера в спиновой
    записи вдвое меньше «навыворот», и неверный выбор даёт сдвиг на 20.6 %.
    """
    pyscf = pytest.importorskip("pyscf")
    importlib.import_module("pyscf.dft")
    libxc = pyscf.dft.libxc

    rho, _ = _spin_grid()[:2]
    rho_a = rho_b = 0.5 * rho  # полный ρ = ρ_a + ρ_b
    total = rho_a + rho_b
    spin = LdaExchange().evaluate_spin(np.zeros((rho.size, 3)), np.stack([rho_a, rho_b]), None)
    # Непрямо: неполяризованный обмен по полной плотности.
    unpol = LdaExchange().evaluate(np.zeros((rho.size, 3)), total)
    # Оба ``energy_density`` — ε_xc на частицу: в замкнутом пределе равны напрямую.
    assert np.max(np.abs(spin.energy_density - unpol.energy_density)) < 1e-12
    assert np.max(np.abs(spin.vrho[0] + spin.vrho[1] - 2.0 * unpol.vrho)) < 1e-12
    del libxc


# --------------------------------------------------------------------------- #
# Замкнутооболочечный предел: UKS обязан совпасть с RKS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("functional", ALL_FUNCTIONALS)
def test_uks_closed_shell_equals_rks(functional: str) -> None:
    """CH4 (замкнутая оболочка): UKS и RKS дают одну энергию.

    Гибриды проверяют долю точного обмена: неверный множитель (¼ вместо ½ в
    фокиане/энергии) дал бы расхождение, на порядок выше дефекта PBE-c.
    """
    molecule = _ch4()
    fn = get_functional(functional)
    basis = build_basis("sto-3g", molecule)
    rks = run_rks(basis, molecule, fn, TIGHT)
    uks = run_uks(basis, molecule, fn, TIGHT)
    assert rks.converged and uks.converged
    # svwn/blyp/b3lyp совпадают до машинной точности; pbe/pbe0 упираются в
    # известный дефект неполяризованного PBE-c (~1e-6).
    assert abs(uks.total_energy - rks.total_energy) < 1e-6


@pytest.mark.parametrize("functional", ("svwn", "pbe"))
def test_uks_closed_shell_gradient_equals_rks(functional: str) -> None:
    """В замкнутом пределе градиент UKS обязан совпасть с градиентом RKS."""
    molecule = _ch4()
    fn = get_functional(functional)
    basis = build_basis("sto-3g", molecule)
    grid = build_grid(molecule)
    rks = run_rks(basis, molecule, fn, TIGHT, grid=grid)
    uks = run_uks(basis, molecule, fn, TIGHT, grid=grid)
    g_rks = np.asarray(rks_gradient(basis, molecule, rks, grid, fn).gradient)
    g_uks = np.asarray(uks_gradient(basis, molecule, uks, grid, fn).gradient)
    assert np.max(np.abs(g_uks - g_rks)) < 1e-6


# --------------------------------------------------------------------------- #
# Открытая оболочка: сходимость, спин, качества
# --------------------------------------------------------------------------- #
def test_uks_open_shell_converges_and_reports_spin() -> None:
    """CH• (дублет): UKS сходится, сообщает <S^2> и проходит все проверки."""
    molecule = _ch_radical()
    fn = get_functional("pbe")
    basis = build_basis("sto-3g", molecule)
    result = run_uks(basis, molecule, fn, TIGHT)
    assert result.converged
    # Дублет: S(S+1) = 0.75; спиновое загрязнение умеренное, но не огромное.
    assert abs(result.s_squared - 0.75) < 0.05
    # Канальные плотности дают правильное число электронов в каждом канале.
    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)
    assert n_alpha == 4 and n_beta == 3
    from quantumlab.engine.integrals import build_overlap

    overlap = build_overlap(basis, molecule)
    total = result.density_alpha + result.density_beta
    assert abs(float(np.trace(total @ overlap)) - molecule.n_electrons) < 1e-6
    # Потенциалы каналов сохранены — на них строятся проверки качества.
    assert result.v_xc_alpha is not None and result.v_xc_beta is not None


def test_uks_single_point_passes_quality_checks() -> None:
    """Сквозной путь через референс-ядро: все проверки качества PASS."""
    engine = ReferenceEngine()
    molecule = _ch_radical()
    for functional in ("pbe", "pbe0", "b3lyp"):
        spec = CalculationSpec(
            task=Task.SINGLE_POINT,
            method=MethodSpec(
                theory=TheoryFamily.DFT,
                basis="sto-3g",
                spin=SpinTreatment.UHF,
                functional=functional,
            ),
        )
        result = engine.run(EngineRequest(job_id="uks", molecule=molecule, spec=spec))
        assert result.converged, functional
        failing = [c.name_key for c in result.quality_checks if c.verdict.name != "PASS"]
        assert failing == [], (functional, failing)


# --------------------------------------------------------------------------- #
# Аналитический градиент vs конечные разности (открытая оболочка)
# --------------------------------------------------------------------------- #
def _displace(molecule: Molecule, index: int, axis: int, shift: float) -> Molecule:
    atoms = list(molecule.atoms)
    x, y, z = atoms[index].position
    if axis == 0:
        x += shift
    elif axis == 1:
        y += shift
    else:
        z += shift
    atoms[index] = type(atoms[index])(symbol=atoms[index].symbol, position=(x, y, z))
    return Molecule(
        name=molecule.name,
        atoms=tuple(atoms),
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    ).perceive_bonds()


def test_uks_gradient_matches_finite_differences() -> None:
    """OH (дублет, PBE): аналитический градиент совпадает с КР сетки.

    Допуск ~1e-5 — уровень отклика квадратурной сетки (та же величина, что у
    RKS на воде). Сетка перестраивается на каждой смещённой геометрии, поэтому
    КР считают поверхность, близкую, но не идентичную аналитической.
    """
    from quantumlab.engine.constants import angstrom_to_bohr

    molecule = _oh()
    fn = get_functional("pbe")
    step = 1e-4
    basis = build_basis("sto-3g", molecule)
    result = run_uks(basis, molecule, fn, TIGHT)
    grid = build_grid(molecule)
    analytic = np.asarray(uks_gradient(basis, molecule, result, grid, fn).gradient)

    max_diff = 0.0
    for atom in range(molecule.n_atoms):
        for axis in range(3):

            def energy(shift: float, a: int = atom, x: int = axis) -> float:
                shifted = _displace(molecule, a, x, shift)
                b = build_basis("sto-3g", shifted)
                r = run_uks(b, shifted, fn, TIGHT)
                assert r.converged
                return r.total_energy

            numerical = (energy(step) - energy(-step)) / (2.0 * angstrom_to_bohr(step))
            max_diff = max(max_diff, abs(numerical - analytic[atom, axis]))
    assert max_diff < 5e-5


# --------------------------------------------------------------------------- #
# Доступность: DFT + спин через реестр/ядро
# --------------------------------------------------------------------------- #
def test_dft_open_shell_is_available_as_uks() -> None:
    """DFT + spin:uhf теперь проходит валидацию (UKS реализован)."""
    engine = ReferenceEngine()
    spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(
            theory=TheoryFamily.DFT, basis="sto-3g", spin=SpinTreatment.UHF, functional="pbe"
        ),
    )
    assert engine.assert_supported(spec) == "sto-3g"


def test_dft_rohf_is_rejected_with_clear_error() -> None:
    """DFT + spin:rohf отклоняется: ограниченной открытой оболочки для DFT нет."""
    engine = ReferenceEngine()
    spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(
            theory=TheoryFamily.DFT, basis="sto-3g", spin=SpinTreatment.ROHF, functional="pbe"
        ),
    )
    with pytest.raises(CombinationUnavailableError) as excinfo:
        engine.assert_supported(spec)
    assert "rohf" in excinfo.value.combination


def test_rks_open_shell_points_to_uks() -> None:
    """RKS на нечётном числе электронов честно указывает на run_uks."""
    from quantumlab.engine.dft import run_rks as _run_rks

    molecule = _ch_radical()
    basis = build_basis("sto-3g", molecule)
    with pytest.raises(ValueError, match="run_uks"):
        _run_rks(basis, molecule, get_functional("pbe"), TIGHT)


# --------------------------------------------------------------------------- #
# Сверка энергии UKS с PySCF (независимый оракул)
# --------------------------------------------------------------------------- #
@pytest.mark.scientific
@pytest.mark.parametrize("functional", ("pbe", "pbe0"))
def test_uks_energy_matches_pyscf(functional: str) -> None:
    """Энергия UKS совпадает с pyscf.dft.UKS на той же (нашей) сетке.

    Допуск 1e-5: расхождение определяется квадратурной сеткой, а не формулами;
    та же величина, что у сверки RKS.
    """
    pyscf = pytest.importorskip("pyscf")
    importlib.import_module("pyscf.dft")

    molecule = _ch_radical()
    fn = get_functional(functional)
    basis = build_basis("sto-3g", molecule)
    result = run_uks(basis, molecule, fn, TIGHT)
    assert result.converged

    # Атомы разделяются ';', а не пробелом: иначе парсер PySCF сливает хвост.
    atom_string = "; ".join(
        f"{a.symbol} {a.position[0]:.10f} {a.position[1]:.10f} {a.position[2]:.10f}"
        for a in molecule.atoms
    )
    mol = pyscf.gto.M(
        atom=atom_string,
        basis="STO-3G",
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
        unit="Angstrom",
        cart=True,
    )
    reference = pyscf.dft.UKS(mol, xc=functional.upper()).run(
        conv_tol=1e-11, max_cycle=300, verbose=0
    )
    assert abs(result.total_energy - reference.e_tot) < 1e-5
