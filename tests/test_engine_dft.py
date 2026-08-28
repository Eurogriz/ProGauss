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
    DispersionCorrection,
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
    Pbe0,
    PbeCorrelation,
    PbeExchange,
    PwCorrelation,
    Svwn,
    VwnCorrelation,
    density_at_points,
    evaluate_basis,
    evaluate_basis_with_gradients,
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
from quantumlab.engine.scf import build_integrals, coulomb_matrix, exchange_matrix, run_rhf
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
    exchange = LdaExchange().evaluate(grid.points, rho)
    from_class = float(np.sum(grid.weights * rho * exchange.energy_density))
    manual = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * float(np.sum(grid.weights * rho ** (4.0 / 3.0)))
    assert from_class == pytest.approx(manual, abs=1e-10)
    assert from_class == pytest.approx(WATER_SLATER_EXCHANGE, abs=1e-6)


def test_functionals_return_zero_for_zero_density() -> None:
    """Нулевая плотность не должна давать NaN: в хвостах молекулы это обычное дело."""
    points = np.zeros((4, 3))
    for functional in (LdaExchange(), VwnCorrelation(), Svwn()):
        xc = functional.evaluate(points, np.zeros(4))
        assert np.all(xc.energy_density == 0.0)
        assert np.all(xc.vrho == 0.0)


def test_lda_exchange_potential_is_numerical_derivative_of_energy() -> None:
    """v_x = d(ρ ε_x)/dρ = 4/3 ε_x — проверка через конечные разности."""
    rho = 0.7
    step = 1e-6
    functional = LdaExchange()
    points = np.zeros((1, 3))
    vxc = functional.evaluate(points, np.array([rho])).vrho
    upper = functional.evaluate(points, np.array([rho + step])).energy_density
    lower = functional.evaluate(points, np.array([rho - step])).energy_density
    numerical = ((rho + step) * upper[0] - (rho - step) * lower[0]) / (2.0 * step)
    assert float(vxc[0]) == pytest.approx(numerical, rel=1e-8)


def test_svwn_declines_spin_polarization() -> None:
    """Спиновая поляризация требует UKS — его нет, и молча считать нельзя."""
    with pytest.raises(NotImplementedError):
        Svwn().evaluate(np.zeros((1, 3)), np.array([1.0]), spin_polarized=True)


def test_get_functional_rejects_unimplemented() -> None:
    """Заявленный в ТЗ функционал без кода отклоняется штатной ошибкой."""
    assert get_functional("svwn").name == "svwn"
    assert get_functional("pbe").name == "pbe"
    with pytest.raises(FunctionalNotFoundError):
        get_functional("b3lyp")


def test_functional_registry_and_capabilities_agree() -> None:
    """Реестр берёт истину из модуля функционалов — расхождение невозможно."""
    registry = default_registry()
    for name in FUNCTIONALS:
        assert registry.availability(f"functional:{name}").is_usable, name
    # Заявленное в ТЗ, но не реализованное — по-прежнему честно недоступно.
    assert not registry.availability("functional:b3lyp").is_usable
    assert not registry.availability("functional:wb97x-d").is_usable


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
    assert any(warning.key == "warning.grid_prune_unimplemented" for warning in result.warnings)


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

    exchange = LdaExchange().evaluate(points, densities)
    reference_exc, reference_vxc, _, _ = libxc.eval_xc("LDA_X", (densities,))
    assert float(np.max(np.abs(exchange.energy_density - reference_exc))) < 1e-12
    assert float(np.max(np.abs(exchange.vrho - reference_vxc))) < 1e-12

    correlation = VwnCorrelation().evaluate(points, densities)
    ref_exc_c, ref_vxc_c, _, _ = libxc.eval_xc("VWN5", (densities,))
    assert float(np.max(np.abs(correlation.energy_density - ref_exc_c))) < 1e-12
    assert float(np.max(np.abs(correlation.vrho - ref_vxc_c))) < 1e-12


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


# --------------------------------------------------------------------------- #
# GGA: PBE
# --------------------------------------------------------------------------- #
@pytest.fixture(name="gradient_points")
def fixture_gradient_points() -> tuple[np.ndarray, np.ndarray]:
    """Плотности и их градиенты для проверки GGA-функционалов."""
    rho = np.array([1.0, 0.5, 0.1, 0.01, 1e-3])
    grad = np.stack(
        [
            np.array([0.4, 0.2, 0.05, 0.01, 1e-3]),
            np.array([0.3, 0.1, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.2, 0.02, 0.0, 0.0]),
        ],
        axis=1,
    )
    return rho, grad


def test_basis_gradients_match_finite_differences(water: Molecule) -> None:
    """Аналитический градиент базисной функции против центральных разностей.

    Проверяется на базисе с p-функциями: у s-гауссиан угловой многочлен тривиален
    и ошибка в его производной осталась бы незамеченной.
    """
    basis = build_basis("6-31g", water)
    rng = np.random.default_rng(7)
    points = rng.normal(size=(40, 3)) * 1.5
    _, gradients = evaluate_basis_with_gradients(basis, water, points)
    step = 1e-5
    worst = 0.0
    for axis in range(3):
        shifted_up = points.copy()
        shifted_up[:, axis] += step
        shifted_down = points.copy()
        shifted_down[:, axis] -= step
        up, _ = evaluate_basis_with_gradients(basis, water, shifted_up)
        down, _ = evaluate_basis_with_gradients(basis, water, shifted_down)
        finite = (up - down) / (2.0 * step)
        worst = max(worst, float(np.max(np.abs(finite - gradients[:, :, axis]))))
    assert worst < 1e-7


def test_gga_functional_refuses_to_run_without_gradient(
    gradient_points: tuple[np.ndarray, np.ndarray],
) -> None:
    """GGA без градиента — не «ноль поправки», а ошибка: результат был бы неверным."""
    rho, _ = gradient_points
    with pytest.raises(ValueError, match="градиент плотности"):
        PbeExchange().evaluate(np.zeros((rho.size, 3)), rho, None)
    with pytest.raises(ValueError, match="градиент плотности"):
        PbeCorrelation().evaluate(np.zeros((rho.size, 3)), rho, None)


def test_pw92_matches_libxc(gradient_points: tuple[np.ndarray, np.ndarray]) -> None:
    """PW92 совпадает с LibXC до машинной точности."""
    libxc = pytest.importorskip("pyscf.dft.libxc", reason="LibXC нужен для независимой сверки")
    rho, _ = gradient_points
    ours = PwCorrelation().evaluate(np.zeros((rho.size, 3)), rho)
    reference = libxc.eval_xc("LDA_C_PW", rho.reshape(1, 1, rho.size))
    assert float(np.max(np.abs(ours.energy_density - reference[0]))) < 1e-14
    assert float(np.max(np.abs(ours.vrho - reference[1][0]))) < 1e-14


def test_pbe_exchange_matches_libxc(gradient_points: tuple[np.ndarray, np.ndarray]) -> None:
    """Обмен PBE совпадает с LibXC до машинной точности по энергии и потенциалам."""
    libxc = pytest.importorskip("pyscf.dft.libxc", reason="LibXC нужен для независимой сверки")
    rho, grad = gradient_points
    ours = PbeExchange().evaluate(np.zeros((rho.size, 3)), rho, grad)
    raw = np.stack([np.vstack([rho, grad[:, 0], grad[:, 1], grad[:, 2]])], axis=0)
    reference_exc, reference_v, _, _ = libxc.eval_xc("GGA_X_PBE", raw)
    assert float(np.max(np.abs(ours.energy_density - reference_exc))) < 1e-14
    assert ours.vsigma is not None
    assert float(np.max(np.abs(ours.vrho - reference_v[0]))) < 1e-14
    assert float(np.max(np.abs(ours.vsigma - reference_v[1]))) < 1e-13


def test_pbe_correlation_potential_is_derivative_of_its_energy(
    gradient_points: tuple[np.ndarray, np.ndarray],
) -> None:
    """v_ρ корреляции PBE совпадает с численной производной её же энергии.

    Регрессия на реальную ошибку: в ∂D/∂A была потеряна степень t², то есть
    стояло u² вместо u³. Энергия этой производной не использует вовсе, поэтому
    расчёт «сходился» и выглядел правдоподобно, а потенциал расходился с
    эталоном на 1.3e-03. Поймать это можно только независимой производной —
    сверка одной энергии ошибку не показывает.
    """
    rho, grad = gradient_points
    functional = PbeCorrelation()
    points = np.zeros((rho.size, 3))
    step = 1e-6
    ours = functional.evaluate(points, rho, grad).vrho
    upper = functional.evaluate(points, rho + step, grad).energy_density
    lower = functional.evaluate(points, rho - step, grad).energy_density
    numerical = ((rho + step) * upper - (rho - step) * lower) / (2.0 * step)
    assert float(np.max(np.abs(numerical - ours))) < 1e-6


def test_pbe_correlation_matches_libxc(gradient_points: tuple[np.ndarray, np.ndarray]) -> None:
    """Корреляция PBE против LibXC с измеренным, а не желаемым допуском.

    Энергия сходится до 2.5e-07, а не до машинной точности: перебор β, γ и
    трёх вариантов базовой LDA-корреляции не дал полного совпадения, и остаток
    остаётся необъяснённым. Он на три порядка ниже химической точности и
    учтён в допуске верификации, но выдавать его за ноль было бы неправдой.
    """
    libxc = pytest.importorskip("pyscf.dft.libxc", reason="LibXC нужен для независимой сверки")
    rho, grad = gradient_points
    ours = PbeCorrelation().evaluate(np.zeros((rho.size, 3)), rho, grad)
    raw = np.stack([np.vstack([rho, grad[:, 0], grad[:, 1], grad[:, 2]])], axis=0)
    reference_exc, reference_v = (
        libxc.eval_xc("GGA_C_PBE", raw)[0],
        libxc.eval_xc("GGA_C_PBE", raw)[1],
    )
    assert float(np.max(np.abs(ours.energy_density - reference_exc))) < 1e-6
    assert ours.vsigma is not None
    assert float(np.max(np.abs(ours.vrho - reference_v[0]))) < 1e-6
    assert float(np.max(np.abs(ours.vsigma - reference_v[1]))) < 1e-4


def test_pbe_energy_matches_pyscf(water: Molecule) -> None:
    """Полная энергия RKS/PBE сверяется с PySCF на воде."""
    pyscf = pytest.importorskip("pyscf", reason="PySCF нужен только для независимой сверки")
    pyscf_dft = pytest.importorskip("pyscf.dft", reason="PySCF DFT нужен для независимой сверки")
    basis = build_basis("sto-3g", water)
    ours = run_rks(basis, water, get_functional("pbe"), grid_preset=GridPreset.ULTRAFINE)

    theirs = pyscf.gto.M(
        atom=[
            ["O", [0.0, 0.0, 0.0]],
            ["H", [0.7571689334, 0.5865799573, 0.0]],
            ["H", [-0.7571689334, 0.5865799573, 0.0]],
        ],
        basis="sto-3g",
        cart=True,
        verbose=0,
    )
    their_scf = pyscf_dft.RKS(theirs)
    their_scf.xc = "PBE"
    their_scf.grids.atom_grid = (120, 974)
    their_scf.verbose = 0
    their_scf.run(conv_tol=1e-12)
    assert ours.total_energy == pytest.approx(float(their_scf.e_tot), abs=5e-6)


def test_engine_refuses_unimplemented_dispersion(water: Molecule) -> None:
    """План с D3(BJ) не должен выполняться как расчёт без поправки (§54 ТЗ).

    До этой проверки движок дисперсию вообще не смотрел: спецификация с
    ``d3bj`` молча считалась как расчёт без неё, и пользователь получал другое
    число под тем же описанием.
    """
    spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(
            theory=TheoryFamily.DFT,
            basis="sto-3g",
            functional="pbe",
            dispersion=DispersionCorrection.D3_BJ,
        ),
    )
    with pytest.raises(MethodNotAvailableError):
        ReferenceEngine().run(EngineRequest(job_id="dft", spec=spec, molecule=water, threads=1))


def test_registry_reports_dispersion_honestly() -> None:
    """Дисперсионные поправки видны в реестре: ``none`` доступна, остальные нет."""
    registry = default_registry()
    assert registry.is_available("dispersion:none")
    assert not registry.is_available("dispersion:d3bj")
    assert not registry.is_available("dispersion:d4")


# --------------------------------------------------------------------------- #
# Гибриды
# --------------------------------------------------------------------------- #
def test_pbe0_declares_its_exact_exchange_share() -> None:
    """PBE0 — гибрид с долей точного обмена ¼ и классом ``hybrid``."""
    functional = Pbe0()
    assert functional.is_hybrid
    assert functional.functional_class == "hybrid"
    assert functional.exact_exchange_fraction == 0.25
    # Доля полунелокального обмена обязана дополнять точную до единицы, иначе
    # обменная энергия не сохранится при переходе LDA → гибрид.
    assert functional.exact_exchange_fraction + functional.dft_exchange_fraction == 1.0


def test_pbe0_scales_only_the_dft_exchange(
    gradient_points: tuple[np.ndarray, np.ndarray],
) -> None:
    """DFT-часть PBE0 — это ¾ обмена PBE плюс полная корреляция PBE."""
    rho, grad = gradient_points
    points = np.zeros((rho.size, 3))
    hybrid = Pbe0().evaluate(points, rho, grad)
    gga_exchange = PbeExchange().evaluate(points, rho, grad)
    gga_correlation = PbeCorrelation().evaluate(points, rho, grad)
    assert np.allclose(
        hybrid.energy_density, 0.75 * gga_exchange.energy_density + gga_correlation.energy_density
    )
    assert hybrid.vsigma is not None and gga_exchange.vsigma is not None
    assert gga_correlation.vsigma is not None
    assert np.allclose(hybrid.vsigma, 0.75 * gga_exchange.vsigma + gga_correlation.vsigma)


def test_hybrid_total_energy_matches_the_iteration_history(water: Molecule) -> None:
    """Возвращённая энергия гибрида равна энергии последней итерации.

    Регрессия на реальную ошибку: обменный член −¼α·D:K добавлялся в энергию
    внутри цикла, но итоговая энергия пересобиралась на сошедшейся плотности
    без него. В результате история итераций показывала одно число, а результат
    — другое, отличающееся ровно на ¼α·D:K (для воды/STO-3G это 2.28 э).
    Сравнение энергии с историей ловит расхождение напрямую.
    """
    basis = build_basis("sto-3g", water)
    result = run_rks(basis, water, Pbe0(), grid_preset=GridPreset.ULTRAFINE)
    assert result.exact_exchange_fraction == 0.25
    assert result.history
    assert result.total_energy == pytest.approx(result.history[-1].energy, abs=1e-9)


def test_hybrid_fock_carries_half_alpha_exchange(water: Molecule) -> None:
    """В фокиане гибрида стоит −½α·K, а не −α·K.

    Из ``E_x^exact = −¼α·D:K`` следует ``∂E/∂D = −½α·K`` — тот же коэффициент ½,
    что у RHF-обмена. Проверка строит оба варианта фокиана и смотрит на
    коммутатор ``FDS − SDF``: при правильном ½ он равен нулю, при «просто α»
    расходится на единицы. Только по энергии это не поймать — SCF сходится и
    с неверным фокианом, просто к другой плотности.
    """
    basis = build_basis("sto-3g", water)
    prepared = build_integrals(basis, water)
    result = run_rks(basis, water, Pbe0(), grid_preset=GridPreset.ULTRAFINE)
    v_xc = result.v_xc
    assert v_xc is not None
    density = result.density
    overlap = prepared.overlap
    alpha = result.exact_exchange_fraction
    exchange = exchange_matrix(density, prepared.eri)
    base = prepared.core + coulomb_matrix(density, prepared.eri) + v_xc - 0.5 * alpha * exchange
    wrong = base - 0.5 * alpha * exchange  # ещё пол-члена, то есть полное −α·K

    def commutator(fock: np.ndarray) -> float:
        return float(np.max(np.abs(fock @ density @ overlap - overlap @ density @ fock)))

    assert commutator(base) < 1e-9
    assert commutator(wrong) > 1e-3


def test_pbe0_energy_matches_pyscf(water: Molecule) -> None:
    """Полная энергия RKS/PBE0 сверяется с PySCF."""
    pyscf = pytest.importorskip("pyscf", reason="PySCF нужен только для независимой сверки")
    pyscf_dft = pytest.importorskip("pyscf.dft", reason="PySCF DFT нужен для независимой сверки")
    basis = build_basis("sto-3g", water)
    ours = run_rks(basis, water, Pbe0(), grid_preset=GridPreset.ULTRAFINE)

    theirs = pyscf.gto.M(
        atom=[
            ["O", [0.0, 0.0, 0.0]],
            ["H", [0.7571689334, 0.5865799573, 0.0]],
            ["H", [-0.7571689334, 0.5865799573, 0.0]],
        ],
        basis="sto-3g",
        cart=True,
        verbose=0,
    )
    their_scf = pyscf_dft.RKS(theirs)
    their_scf.xc = "PBE0"
    their_scf.grids.atom_grid = (120, 974)
    their_scf.verbose = 0
    their_scf.run(conv_tol=1e-12)
    assert ours.total_energy == pytest.approx(float(their_scf.e_tot), abs=5e-6)


def test_exact_exchange_deepens_the_homo(water: Molecule) -> None:
    """Точный обмен углубляет ВЗМО — физическая проверка знака доли обмена.

    Если бы доля вошла с неверным знаком, SCF всё равно сошёлся бы, но ВЗМО
    ушла бы вверх, а не вниз. Это независимый признак корректности, который
    не сводится к сравнению чисел с эталоном.
    """
    basis = build_basis("sto-3g", water)
    gga = run_rks(basis, water, get_functional("pbe"), grid_preset=GridPreset.FINE)
    hybrid = run_rks(basis, water, Pbe0(), grid_preset=GridPreset.FINE)
    gga_homo = gga.orbital_energies[water.n_electrons // 2 - 1]
    hybrid_homo = hybrid.orbital_energies[water.n_electrons // 2 - 1]
    assert hybrid_homo < gga_homo
