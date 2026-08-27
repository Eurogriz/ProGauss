"""Тесты DFT-среза: квадратура, LDA-функционалы, RKS и честность отказов.

Тесты устроены так же, как остальные в проекте: сначала проверяется
математика по частям (каждая часть сетки и каждый функционал отдельно), затем
числа сверяются с независимым оракулом (PySCF/LibXC), и отдельно проверяется,
что движок честно отклоняет то, чего не умеет.

Два последних теста — регрессионные на реальные ошибки, найденные при
разработке: потерянный якобиан ``r²`` в весах сетки и подмена ``E_xc`` следом
``D·V_xc`` в выражении для энергии. Они проверяют не сами ошибки, а то, что
проверки качества **способны их поймать**: контроль, который ни разу не
сработал, ничего не контролирует.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.result import QualityVerdict
from quantumlab.domain.spec import (
    CalculationSpec,
    GridPreset,
    GridSpec,
    MethodSpec,
    OptimizationSpec,
    Task,
    TheoryFamily,
)
from quantumlab.engine import reference as reference_module
from quantumlab.engine.basis import build_basis
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.dft import run_rks
from quantumlab.engine.functional import (
    FUNCTIONALS,
    LdaExchange,
    Svwn,
    VwnCorrelation,
    density_at_points,
    evaluate_basis,
    get_functional,
)
from quantumlab.engine.quadrature import (
    QuadratureGrid,
    angular_grid,
    becke_weights,
    build_grid,
    radial_grid,
)
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.registry import default_registry
from quantumlab.engine.scf import build_integrals, run_rhf
from quantumlab.errors import FunctionalNotFoundError, MethodNotAvailableError

WATER = Path(__file__).parent / "fixtures" / "water.xyz"
HYDROGEN = Path(__file__).parent / "fixtures" / "hydrogen.xyz"

#: Энергия RKS/SVWN воды в STO-3G на сетке ultrafine. Сверена с PySCF 2.14.0
#: (``dft.RKS``, ``xc='LDA,VWN'``): расхождение 4.0e-08 Eh.
WATER_SVWN_ULTRAFINE = -74.7320493297

#: Обменная энергия Слейтера на плотности RHF/STO-3G воды, э.
WATER_SLATER_EXCHANGE = -8.2005509


@pytest.fixture(name="water")
def fixture_water() -> Molecule:
    """Молекула воды из фикстуры."""
    return Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")


@pytest.fixture(name="hydrogen")
def fixture_hydrogen() -> Molecule:
    """Молекула водорода из фикстуры."""
    return Molecule.from_xyz(HYDROGEN.read_text(encoding="utf-8"), name="hydrogen")


# --------------------------------------------------------------------------- #
# Квадратура: каждая часть сетки проверяется сама по себе
# --------------------------------------------------------------------------- #
def test_radial_grid_integrates_hydrogenic_radial_tail() -> None:
    """∫₀^∞ r²e^(−2r) dr = 1/4 — точный ответ, годный для любой сетки."""
    radii, weights = radial_grid(150, alpha=0.8)
    value = float(np.sum(weights * radii**2 * np.exp(-2.0 * radii)))
    assert value == pytest.approx(0.25, abs=1e-12)


def test_radial_grid_is_positive_and_ordered() -> None:
    """Отрицательный вес или неупорядоченные узлы означали бы сломанное отображение."""
    radii, weights = radial_grid(75, alpha=0.8)
    assert np.all(weights > 0.0)
    assert np.all(np.diff(radii) > 0.0)


@pytest.mark.parametrize("preset", list(GridPreset))
def test_angular_weights_reproduce_solid_angle(preset: GridPreset) -> None:
    """Σ w = 4π, ∫cos²θ dΩ = 4π/3, ∫cos θ dΩ = 0 — базовые тождества сферы."""
    n_theta, n_phi = {
        GridPreset.COARSE: (12, 24),
        GridPreset.FINE: (24, 48),
        GridPreset.ULTRAFINE: (32, 64),
    }[preset]
    directions, weights = angular_grid(n_theta, n_phi)
    assert float(np.sum(weights)) == pytest.approx(4.0 * np.pi, abs=1e-13)
    cos_theta = directions[:, 2]
    assert float(np.sum(weights * cos_theta**2)) == pytest.approx(4.0 * np.pi / 3.0, abs=1e-13)
    assert float(np.sum(weights * cos_theta)) == pytest.approx(0.0, abs=1e-13)


def test_becke_weights_partition_unity(water: Molecule) -> None:
    """Σ_A w_A(r) = 1 в каждой точке — иначе плотность считалась бы с ошибкой."""
    grid = build_grid(water, GridPreset.COARSE)
    centers = np.array([atom.position for atom in water.atoms], dtype=float)
    total = np.zeros(grid.n_points)
    for index in range(len(water.atoms)):
        total += becke_weights(grid.points, centers, np.full(grid.n_points, index))
    assert float(np.max(np.abs(total - 1.0))) < 1e-12


@pytest.mark.parametrize(
    ("preset", "tolerance"),
    [(GridPreset.COARSE, 1e-4), (GridPreset.FINE, 1e-6), (GridPreset.ULTRAFINE, 1e-7)],
)
def test_grid_integrates_hf_density_to_electron_count(
    water: Molecule, preset: GridPreset, tolerance: float
) -> None:
    """∫ρ dV по сетке обязано давать число электронов.

    Именно этот контроль ловит ошибку в мере интегрирования: без якобиана r²
    интеграл расходится на десятки единиц и никакая сходимость SCF этого не
    показывает, потому что SCF о сетке не знает.
    """
    basis = build_basis("sto-3g", water)
    rhf = run_rhf(basis, water)
    grid = build_grid(water, preset)
    rho = density_at_points(evaluate_basis(basis, water, grid.points), rhf.density)
    assert float(np.sum(grid.weights * rho)) == pytest.approx(water.n_electrons, abs=tolerance)


def test_grid_remembers_its_preset(water: Molecule) -> None:
    """Пресет хранится в сетке: результат DFT без указания плотности невоспроизводим."""
    assert build_grid(water, GridPreset.ULTRAFINE).preset is GridPreset.ULTRAFINE


# --------------------------------------------------------------------------- #
# Функционалы
# --------------------------------------------------------------------------- #
def test_slater_exchange_matches_analytic_formula(water: Molecule) -> None:
    """E_x = −¾(3/π)^(1/3) ∫ρ^(4/3) dV — независимая от класса формула."""
    basis = build_basis("sto-3g", water)
    rhf = run_rhf(basis, water)
    grid = build_grid(water, GridPreset.ULTRAFINE)
    rho = density_at_points(evaluate_basis(basis, water, grid.points), rhf.density)
    exc, _ = LdaExchange().evaluate(grid.points, rho)
    from_class = float(np.sum(grid.weights * rho * exc))
    manual = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * float(np.sum(grid.weights * rho ** (4.0 / 3.0)))
    assert from_class == pytest.approx(manual, abs=1e-10)
    assert from_class == pytest.approx(WATER_SLATER_EXCHANGE, abs=1e-6)


def test_functionals_return_zero_for_zero_density() -> None:
    """Нулевая плотность не должна давать NaN: в хвостах молекулы это обычное дело."""
    points = np.zeros((4, 3))
    for functional in (LdaExchange(), VwnCorrelation(), Svwn()):
        exc, vxc = functional.evaluate(points, np.zeros(4))
        assert np.all(exc == 0.0)
        assert np.all(vxc == 0.0)


def test_lda_exchange_potential_is_numerical_derivative_of_energy() -> None:
    """v_x = d(ρ ε_x)/dρ = 4/3 ε_x — проверка через конечные разности."""
    rho = 0.7
    step = 1e-6
    functional = LdaExchange()
    _, vxc = functional.evaluate(np.zeros((1, 3)), np.array([rho]))
    upper, _ = functional.evaluate(np.zeros((1, 3)), np.array([rho + step]))
    lower, _ = functional.evaluate(np.zeros((1, 3)), np.array([rho - step]))
    numerical = ((rho + step) * upper[0] - (rho - step) * lower[0]) / (2.0 * step)
    assert float(vxc[0]) == pytest.approx(numerical, rel=1e-8)


def test_svwn_declines_spin_polarization() -> None:
    """Спиновая поляризация требует UKS — его нет, и молча считать нельзя."""
    with pytest.raises(NotImplementedError):
        Svwn().evaluate(np.zeros((1, 3)), np.array([1.0]), spin_polarized=True)


def test_get_functional_rejects_unimplemented() -> None:
    """Заявленный в ТЗ функционал без кода отклоняется штатной ошибкой."""
    assert get_functional("svwn").name == "svwn"
    with pytest.raises(FunctionalNotFoundError):
        get_functional("b3lyp")


def test_functional_registry_and_capabilities_agree() -> None:
    """Реестр берёт истину из модуля функционалов — расхождение невозможно."""
    registry = default_registry()
    for name in FUNCTIONALS:
        assert registry.availability(f"functional:{name}").is_usable
    assert not registry.availability("functional:pbe").is_usable


# --------------------------------------------------------------------------- #
# RKS
# --------------------------------------------------------------------------- #
def test_rks_water_converges_to_reference_value(water: Molecule) -> None:
    """Энергия RKS/SVWN воды совпадает со значением, сверенным с PySCF."""
    basis = build_basis("sto-3g", water)
    result = run_rks(basis, water, get_functional("svwn"), grid_preset=GridPreset.ULTRAFINE)
    assert result.converged
    assert result.total_energy == pytest.approx(WATER_SVWN_ULTRAFINE, abs=1e-6)


def test_rks_improves_monotonically_with_grid_density(water: Molecule) -> None:
    """Сходимость по сетке: расхождение с пределом падает с ростом числа точек."""
    basis = build_basis("sto-3g", water)
    energies = {
        preset: run_rks(basis, water, get_functional("svwn"), grid_preset=preset).total_energy
        for preset in (GridPreset.COARSE, GridPreset.FINE, GridPreset.ULTRAFINE)
    }
    limit = energies[GridPreset.ULTRAFINE]
    assert abs(energies[GridPreset.COARSE] - limit) > abs(energies[GridPreset.FINE] - limit)
    assert abs(energies[GridPreset.FINE] - limit) > 1e-9


def test_rks_rejects_odd_electron_count() -> None:
    """RKS предполагает замкнутую оболочку; нечётное число электронов — отказ."""
    radical = Molecule.from_xyz(
        (Path(__file__).parent / "fixtures" / "ch-radical.xyz").read_text(encoding="utf-8"),
        name="ch",
        multiplicity=2,
    )
    basis = build_basis("sto-3g", radical)
    with pytest.raises(ValueError, match="чётного числа электронов"):
        run_rks(basis, radical, get_functional("svwn"))


def test_rks_reuses_supplied_grid_and_integrals(water: Molecule) -> None:
    """Переданная сетка используется как есть — иначе проверки шли бы по другой сетке."""
    basis = build_basis("sto-3g", water)
    grid = build_grid(water, GridPreset.COARSE)
    result = run_rks(basis, water, get_functional("svwn"), grid=grid)
    assert result.grid_points == grid.n_points


# --------------------------------------------------------------------------- #
# Поведение движка
# --------------------------------------------------------------------------- #
def _dft_spec(
    functional: str | None = "svwn", preset: GridPreset = GridPreset.FINE
) -> CalculationSpec:
    """Спецификация одноточечного RKS-расчёта."""
    return CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=TheoryFamily.DFT, basis="sto-3g", functional=functional),
        grid=GridSpec(preset=preset),
    )


def test_engine_runs_dft_and_reports_all_checks_passing(water: Molecule) -> None:
    """Полный проход через движок: энергия, свойства и все проверки качества."""
    result = ReferenceEngine().run(
        EngineRequest(job_id="dft", spec=_dft_spec(), molecule=water, threads=1)
    )
    assert result.energy_hartree == pytest.approx(-74.7320495103, abs=1e-8)
    assert result.scf_iterations > 1
    assert result.dipole_debye == pytest.approx(1.728787, abs=1e-5)
    failures = [c for c in result.quality_checks if c.verdict is QualityVerdict.FAIL]
    assert failures == []
    keys = {c.name_key for c in result.quality_checks}
    assert "quadrature_electron_count" in keys


def test_engine_warns_that_pruning_is_not_implemented(water: Molecule) -> None:
    """``grid.prune`` объявлен в спецификации, но не реализован — это видно (§54)."""
    result = ReferenceEngine().run(
        EngineRequest(job_id="dft", spec=_dft_spec(), molecule=water, threads=1)
    )
    assert any("Прореживание" in warning for warning in result.warnings)


def test_engine_refuses_dft_without_explicit_functional(water: Molecule) -> None:
    """DFT без функционала отклоняется на уровне спецификации, а не подстановкой.

    Молчаливый выбор «посчитать чем-нибудь по умолчанию» сделал бы результат
    невоспроизводимым для того, кто читает только число (§54 ТЗ).
    """
    with pytest.raises(ValidationError, match="обменно-корреляционный функционал"):
        _dft_spec(functional=None)
    assert water.n_electrons == 10


def test_engine_refuses_dft_geometry_optimization(water: Molecule) -> None:
    """XC-вклада в аналитический градиент нет — оптимизация отклоняется (§54)."""
    spec = CalculationSpec(
        task=Task.OPTIMIZATION,
        method=MethodSpec(theory=TheoryFamily.DFT, basis="sto-3g", functional="svwn"),
        optimization=OptimizationSpec(),
    )
    with pytest.raises(MethodNotAvailableError):
        ReferenceEngine().run(EngineRequest(job_id="dft", spec=spec, molecule=water, threads=1))


def test_engine_refuses_unimplemented_functional(water: Molecule) -> None:
    """B3LYP заявлен в ТЗ, но не реализован — честный отказ вместо числа."""
    with pytest.raises(FunctionalNotFoundError):
        ReferenceEngine().run(
            EngineRequest(
                job_id="dft", spec=_dft_spec(functional="b3lyp"), molecule=water, threads=1
            )
        )


def test_registry_reports_dft_as_partial_with_limitations() -> None:
    """DFT работает, но не всё: ограничения видны в реестре, а не только в коде."""
    capability = default_registry().get("method:dft")
    assert capability.availability.is_usable
    assert len(capability.limitations) >= 3


# --------------------------------------------------------------------------- #
# Регрессии: проверки качества обязаны ловить известные ошибки
# --------------------------------------------------------------------------- #
def test_quadrature_check_catches_missing_jacobian(
    water: Molecule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сетка без якобиана r² должна дать FAIL, а не правдоподобное число.

    Реальная ошибка разработки: радиальные веса — это мера ``dr``, угловые
    содержат ``sin θ``, а множитель ``r²`` из ``dV = r² dr dΩ`` был потерян.
    Энергия получилась −1993 вместо −74.73, и только эта проверка показала
    причину: интеграл плотности дал 237.65 вместо 10.
    """

    def without_jacobian(
        molecule: Molecule, preset: GridPreset, alpha: float = 0.8
    ) -> QuadratureGrid:
        grid = build_grid(molecule, preset, alpha)
        centers = np.array([atom.position for atom in molecule.atoms], dtype=float)
        radius = np.linalg.norm(grid.points - centers[grid.atom_index], axis=1)
        return replace(grid, weights=grid.weights / np.maximum(radius**2, 1e-30))

    monkeypatch.setattr(reference_module, "build_grid", without_jacobian)
    result = ReferenceEngine().run(
        EngineRequest(job_id="dft", spec=_dft_spec(), molecule=water, threads=1)
    )
    check = next(c for c in result.quality_checks if c.name_key == "quadrature_electron_count")
    assert check.verdict is QualityVerdict.FAIL


def test_decomposition_check_catches_exchange_double_counting(water: Molecule) -> None:
    """Подмена E_xc следом D·V_xc должна дать FAIL в разложении энергии.

    Вторая реальная ошибка: ``V_xc`` входит в фокиан, а в энергию — только
    ``E_xc``. Коэффициента ½, как у обмена HF, здесь нет: функционал нелинеен по
    плотности. Прослеженная величина для воды/STO-3G равна −11.69 вместо −8.88.
    """
    basis = build_basis("sto-3g", water)
    grid = build_grid(water, GridPreset.FINE)
    result = run_rks(basis, water, get_functional("svwn"), grid=grid)
    assert result.v_xc is not None
    trace = float(np.sum(result.density * result.v_xc))
    broken = replace(result, total_energy=result.total_energy + trace)
    checks = reference_module._quality_checks_rks(
        broken,
        basis,
        water,
        build_integrals(basis, water),
        grid,
        evaluate_basis(basis, water, grid.points),
    )
    decomposition = next(c for c in checks if c.name_key == "energy_decomposition")
    assert decomposition.verdict is QualityVerdict.FAIL


def test_functionals_match_libxc_oracle() -> None:
    """Сверка обменной и корреляционной частей с LibXC через PySCF."""
    libxc = pytest.importorskip(
        "pyscf.dft.libxc", reason="LibXC нужен только для независимой сверки"
    )
    densities = np.array([5.0, 1.0, 0.5, 0.1, 0.01, 1e-3])
    points = np.zeros((densities.size, 3))

    exc, vxc = LdaExchange().evaluate(points, densities)
    reference_exc, reference_vxc, _, _ = libxc.eval_xc("LDA_X", (densities,))
    assert float(np.max(np.abs(exc - reference_exc))) < 1e-12
    assert float(np.max(np.abs(vxc - reference_vxc))) < 1e-12

    exc_c, vxc_c = VwnCorrelation().evaluate(points, densities)
    ref_exc_c, ref_vxc_c, _, _ = libxc.eval_xc("VWN5", (densities,))
    assert float(np.max(np.abs(exc_c - ref_exc_c))) < 1e-12
    assert float(np.max(np.abs(vxc_c - ref_vxc_c))) < 1e-12


@pytest.mark.parametrize(
    ("fixture_path", "basis_name", "atoms", "reference_energy"),
    [
        (
            WATER,
            "sto-3g",
            "O 0 0 0; H 0.7571689334 0.5865799573 0; H -0.7571689334 0.5865799573 0",
            -74.7320493697,
        ),
        (HYDROGEN, "sto-3g", "H 0 0 0; H 0 0 0.95", -1.0953471114),
        (
            WATER,
            "6-31g",
            "O 0 0 0; H 0.7571689334 0.5865799573 0; H -0.7571689334 0.5865799573 0",
            -75.8179300842,
        ),
    ],
)
def test_rks_energy_matches_pyscf(
    fixture_path: Path, basis_name: str, atoms: str, reference_energy: float
) -> None:
    """Энергия RKS/SVWN сверяется с PySCF на трёх независимых системах."""
    # ``pyscf.dft`` не подтягивается вместе с ``pyscf`` — импортируем явно.
    pyscf = pytest.importorskip("pyscf", reason="PySCF нужен только для независимой сверки")
    pyscf_dft = pytest.importorskip("pyscf.dft", reason="PySCF DFT нужен для независимой сверки")
    molecule = Molecule.from_xyz(fixture_path.read_text(encoding="utf-8"), name=fixture_path.stem)
    basis = build_basis(basis_name, molecule)
    result = run_rks(basis, molecule, get_functional("svwn"), grid_preset=GridPreset.ULTRAFINE)

    spec = [
        [part[0], [float(value) for value in part[1:]]]
        for part in (entry.split() for entry in atoms.split(";"))
    ]
    theirs = pyscf.gto.M(atom=spec, basis=basis_name, cart=True, verbose=0)
    their_scf = pyscf_dft.RKS(theirs)
    their_scf.xc = "LDA,VWN"
    their_scf.grids.atom_grid = (120, 974)
    their_scf.verbose = 0
    their_scf.run(conv_tol=1e-12)
    assert float(their_scf.e_tot) == pytest.approx(reference_energy, abs=1e-8)
    assert result.total_energy == pytest.approx(float(their_scf.e_tot), abs=1e-6)
