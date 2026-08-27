"""Тесты референсного ядра: RHF в точке плюс честность отказов.

Здесь проверяются две вещи, которые одинаково важны:

1. **числа** — энергия, орбитали и диполь совпадают со значениями, независимо
   подтверждёнными сверкой с PySCF (``tests/test_crosscheck_pyscf.py``);
2. **честность** — неподдерживаемая комбинация задачи/метода/спина/базиса
   отклоняется штатной ошибкой, а не превращается в правдоподобное число (§54).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.domain.result import CalculationResult, QualityVerdict
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.reference import ENGINE_BACKEND, ENGINE_NAME, ReferenceEngine
from quantumlab.errors import BasisNotFoundError, MethodNotAvailableError

WATER = Path(__file__).parent / "fixtures" / "water.xyz"

#: Энергия RHF воды в STO-3G, независимо подтверждённая сверкой с PySCF
#: (расхождение 9.0e-08 Eh, см. tests/test_crosscheck_pyscf.py).
WATER_STO3G_ENERGY = -74.9630296563

#: Диполь воды в STO-3G, дебай: совпадает с PySCF до 1e-7 a.u.
WATER_STO3G_DIPOLE = 1.7253


class _Collector:
    """Приёмник прогресса: проверяем, что ядро сообщает о стадиях."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.percent: list[float] = []
        self.extras: list[dict[str, object]] = []

    def report(self, percent: float, stage_key: str, **extra: object) -> bool:
        """Записывает стадию; всегда готов продолжать."""
        self.stages.append(stage_key)
        self.percent.append(percent)
        self.extras.append(dict(extra))
        return True


def _water() -> Molecule:
    return Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")


def _spec(basis: str, *, theory: TheoryFamily = TheoryFamily.HF) -> CalculationSpec:
    return CalculationSpec(task=Task.SINGLE_POINT, method=MethodSpec(theory=theory, basis=basis))


def _run(
    basis: str = "sto-3g", *, molecule: Molecule | None = None, spec: CalculationSpec | None = None
) -> CalculationResult:
    """Запускает ядро с заданной спецификацией и возвращает результат."""
    engine = ReferenceEngine()
    return engine.run(
        EngineRequest(
            job_id="job-test",
            molecule=molecule or _water(),
            spec=spec or _spec(basis),
        )
    )


# --------------------------------------------------------------------------- #
# Числа
# --------------------------------------------------------------------------- #
def test_supported_tasks_is_single_point_only() -> None:
    """Ядро заявляет ровно то, что умеет: только расчёт в точке."""
    assert list(ReferenceEngine().supported_tasks()) == ["single_point"]
    assert ReferenceEngine().name == ENGINE_NAME


def test_water_sto3g_energy_matches_the_verified_value() -> None:
    """Энергия совпадает со значением, сверенным с независимым пакетом."""
    result = _run()
    assert result.energy_hartree == pytest.approx(WATER_STO3G_ENERGY, abs=1e-8)
    assert result.converged
    assert result.scf_iterations == 8


def test_orbital_energies_homo_lumo_and_gap() -> None:
    """Орбитали, границы ЗМО/НСМО и заселенности согласованы с числом электронов."""
    result = _run()
    assert len(result.orbitals) == 7  # STO-3G воды: 7 базисных функций
    assert [orbital.occupation for orbital in result.orbitals] == [2.0] * 5 + [0.0] * 2
    assert result.homo_energy_hartree == pytest.approx(-0.391243, abs=1e-6)
    assert result.lumo_energy_hartree == pytest.approx(0.605165, abs=1e-6)
    assert result.gap_hartree == pytest.approx(0.996408, abs=1e-6)
    # ЗМО обязана быть ниже НСМО, иначе «щель» не имеет смысла.
    homo = result.homo_energy_hartree
    lumo = result.lumo_energy_hartree
    assert homo is not None and lumo is not None
    assert homo < lumo
    assert result.gap_hartree == pytest.approx(lumo - homo, abs=1e-12)


def test_dipole_matches_the_verified_value() -> None:
    """Диполь в дебаях совпадает с PySCF (см. кросс-проверку)."""
    result = _run()
    assert result.dipole_debye == pytest.approx(WATER_STO3G_DIPOLE, abs=1e-4)


def test_quality_checks_pass_on_a_cartesian_basis() -> None:
    """Все проверки качества проходят для базиса в декартовой схеме."""
    result = _run()
    verdicts = {check.name_key: check.verdict for check in result.quality_checks}
    assert verdicts["scf_converged"] is QualityVerdict.PASS
    assert verdicts["electron_count"] is QualityVerdict.PASS
    assert verdicts["density_idempotency"] is QualityVerdict.PASS
    assert verdicts["basis_angular_scheme"] is QualityVerdict.PASS
    # Теорема вириала на неоптимизированной геометрии выполняется приближённо.
    assert verdicts["virial_ratio"] in (QualityVerdict.PASS, QualityVerdict.WARNING)
    assert not result.warnings


def test_electron_count_check_detects_the_true_electron_number() -> None:
    """tr(D·S) равен числу электронов молекулы, а не чему-то похожему."""
    result = _run()
    check = next(c for c in result.quality_checks if c.name_key == "electron_count")
    assert check.detail is not None
    assert "10.00000000" in check.detail


def test_progress_is_reported_for_every_stage() -> None:
    """Прогресс сообщается по стадиям и монотонно растёт до 100%."""
    collector = _Collector()
    ReferenceEngine().run(
        EngineRequest(job_id="job-test", molecule=_water(), spec=_spec("sto-3g")),
        progress=collector,
    )
    assert collector.stages == ["basis", "integrals", "scf", "properties"]
    assert collector.percent == sorted(collector.percent)
    assert collector.percent[-1] == pytest.approx(100.0)
    # Стадии несут полезный контекст: число функций, число итераций SCF.
    assert "functions" in collector.extras[0]
    assert "iterations" in collector.extras[2]


def test_result_is_reproducible_and_carries_environment() -> None:
    """Два одинаковых запуска дают одинаковый отпечаток и полное окружение."""
    first = _run()
    second = _run()
    assert first.fingerprint.digest == second.fingerprint.digest
    assert first.energy_hartree == second.energy_hartree

    environment = first.environment
    assert environment.engine_backend == ENGINE_BACKEND
    assert environment.software_version
    assert environment.hostname
    assert environment.cores >= 1
    assert environment.mpi_ranks == 1

    stages = [record.stage for record in first.timings]
    assert stages == ["basis", "integrals", "scf", "properties"]
    assert all(record.wall_seconds >= 0.0 for record in first.timings)


def test_energies_are_variationally_ordered_by_basis_size() -> None:
    """Больший базис даёт меньшую энергию: вариационный принцип в действии."""
    small = _run("sto-3g")
    larger = _run("6-31g")
    assert larger.energy_hartree < small.energy_hartree


# --------------------------------------------------------------------------- #
# Честность отказов
# --------------------------------------------------------------------------- #
def test_spherical_basis_runs_but_warns_about_the_scheme() -> None:
    """Базис со сферической публикацией d считается — но с явным предупреждением.

    Мы не блокируем расчёт и не делаем вид, что результат совпадает с
    табличным: разница схемы отражена и в проверке качества, и в warnings.

    Молекула взята наименьшая из возможных (H2): предупреждение зависит от
    базиса, а не от системы, а стоимость ERI растёт как четвёртая степень
    числа функций.
    """
    hydrogen = Molecule(
        name="h2",
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, 0.7414)),
        ),
    )
    result = _run("cc-pvdz", molecule=hydrogen)
    assert result.converged
    check = next(c for c in result.quality_checks if c.name_key == "basis_angular_scheme")
    assert check.verdict is QualityVerdict.WARNING
    assert check.detail is not None and "сферической" in check.detail
    assert len(result.warnings) == 1
    assert "сферической" in result.warnings[0]


def test_unsupported_task_is_rejected_before_any_computation() -> None:
    """Оптимизация требует градиентов, которых нет — задача отклоняется."""
    with pytest.raises(MethodNotAvailableError):
        _run(
            spec=CalculationSpec(
                task=Task.OPTIMIZATION,
                method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
            )
        )


def test_unimplemented_theory_is_rejected() -> None:
    """DFT, MP2 и CC не реализованы — их нельзя получить даже формально.

    Отклонение идёт по методу, а не по функционалу: DFT без функционала
    ``MethodSpec`` не построит вовсе, поэтому для него функционал указан.
    """
    for theory in (TheoryFamily.MP2, TheoryFamily.SCS_MP2, TheoryFamily.CCSD, TheoryFamily.CCSD_T):
        with pytest.raises(MethodNotAvailableError):
            _run(
                spec=CalculationSpec(
                    task=Task.SINGLE_POINT,
                    method=MethodSpec(theory=theory, basis="sto-3g"),
                )
            )
    with pytest.raises(MethodNotAvailableError):
        _run(
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=MethodSpec(theory=TheoryFamily.DFT, basis="sto-3g", functional="pbe0"),
            )
        )


def test_unsupported_spin_treatment_is_rejected() -> None:
    """UHF/ROHF не реализованы: нечётная система не должна «считаться» как RHF."""
    hydrogen = Molecule(
        name="h", atoms=(Atom(symbol="H", position=(0.0, 0.0, 0.0)),), multiplicity=2
    )
    with pytest.raises(MethodNotAvailableError):
        _run(
            molecule=hydrogen,
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.UHF),
            ),
        )


def test_missing_method_spec_is_rejected_with_an_explanation() -> None:
    """Без явного метода расчёт не запускается.

    Молчаливый выбор базиса «по умолчанию» сделал бы результат
    невоспроизводимым, поэтому вместо догадки — внятная ошибка.
    """
    with pytest.raises(ValueError, match=r"spec\.method"):
        _run(spec=CalculationSpec(task=Task.SINGLE_POINT))


def test_unknown_basis_is_reported_as_missing_not_as_unavailable() -> None:
    """Неизвестный базис — это «не найден», а не «недоступен»: разные действия UI."""
    with pytest.raises(BasisNotFoundError):
        _run(
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=MethodSpec(theory=TheoryFamily.HF, basis="unobtainium-qzvp"),
            )
        )


def test_assert_supported_returns_the_basis_it_validated() -> None:
    """Предпроверка возвращает базис — её используют CLI, GUI и REST."""
    engine = ReferenceEngine()
    assert engine.assert_supported(_spec("sto-3g")) == "sto-3g"
    with pytest.raises(MethodNotAvailableError):
        engine.assert_supported(CalculationSpec(task=Task.IRC))
