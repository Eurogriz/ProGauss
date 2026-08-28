"""Тесты референсного ядра: RHF в точке плюс честность отказов.

Здесь проверяются две вещи, которые одинаково важны:

1. **числа** — энергия, орбитали и диполь совпадают со значениями, независимо
   подтверждёнными сверкой с PySCF (``tests/test_crosscheck_pyscf.py``);
2. **честность** — неподдерживаемая комбинация задачи/метода/спина/базиса
   отклоняется штатной ошибкой, а не превращается в правдоподобное число (§54).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.domain.result import CalculationResult, QualityVerdict
from quantumlab.domain.spec import (
    CalculationSpec,
    CoordinateConstraint,
    MethodSpec,
    OptimizationSpec,
    ScfSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.reference import (
    ENGINE_BACKEND,
    ENGINE_NAME,
    ReferenceEngine,
    _scf_settings,
)
from quantumlab.engine.registry import default_registry
from quantumlab.engine.scf import ScfSettings, run_rhf
from quantumlab.errors import (
    BasisNotFoundError,
    FunctionalNotFoundError,
    MethodNotAvailableError,
)

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
def test_supported_tasks_list_only_implemented_ones() -> None:
    """Ядро заявляет ровно то, что умеет: точка, оптимизация и частоты."""
    assert list(ReferenceEngine().supported_tasks()) == [
        "single_point",
        "optimization",
        "frequencies",
    ]
    assert ReferenceEngine().name == ENGINE_NAME


def test_tasks_without_a_kernel_are_not_claimed() -> None:
    """Задачи без ядра не заявлены и отклоняются до начала вычислений.

    Частоты в этот список больше не входят: гессиан из аналитических градиентов
    реализован. Примером недоступной задачи служит IRC — для него нужен путь по
    дну долины, которого нет (§54 ТЗ).
    """
    supported = list(ReferenceEngine().supported_tasks())
    assert "frequencies" in supported
    assert "irc" not in supported
    with pytest.raises(MethodNotAvailableError):
        ReferenceEngine().assert_supported(
            CalculationSpec(
                task=Task.IRC,
                method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
            )
        )


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
    # Оба — точные тождества, а не приближённые критерии (см. test_scf.py:
    # отношение −V/T критерием быть не может в конечном базисе).
    assert verdicts["energy_decomposition"] is QualityVerdict.PASS
    assert verdicts["fock_density_commutator"] is QualityVerdict.PASS
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
    # Проверка по ключу, а не по тексту: предупреждение локализуется, и
    # привязывать тест к русскому варианту значило бы сломать его при
    # переключении языка.
    assert result.warnings[0].key == "warning.basis_spherical_scheme"
    assert result.warnings[0].params["basis"] == "cc-pvdz"


def test_unsupported_task_is_rejected_before_any_computation() -> None:
    """Задача без ядра отклоняется до начала вычислений.

    Прежде этот тест проверял отклонение **оптимизации** и проходил лишь потому,
    что ``OptimizationSpec`` по умолчанию просил ``redundant_internal``. Градиенты
    и оптимизация давно реализованы, поэтому примером служит IRC — для него нет
    ни гессиана вдоль пути, ни самого пути.
    """
    with pytest.raises(MethodNotAvailableError):
        _run(
            spec=CalculationSpec(
                task=Task.IRC,
                method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
            )
        )


def test_unimplemented_theory_is_rejected() -> None:
    """MP2 и CC не реализованы — их нельзя получить даже формально.

    DFT из этого списка исключён: LDA-функционал SVWN реализован и считается
    (см. tests/test_engine_dft.py). Отклоняется конкретный нереализованный
    функционал, причём отдельным типом ошибки — так GUI может показать
    «функционал недоступен» вместо общего «метод недоступен».
    """
    for theory in (TheoryFamily.MP2, TheoryFamily.SCS_MP2, TheoryFamily.CCSD, TheoryFamily.CCSD_T):
        with pytest.raises(MethodNotAvailableError):
            _run(
                spec=CalculationSpec(
                    task=Task.SINGLE_POINT,
                    method=MethodSpec(theory=theory, basis="sto-3g"),
                )
            )
    with pytest.raises(FunctionalNotFoundError):
        _run(
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=MethodSpec(theory=TheoryFamily.DFT, basis="sto-3g", functional="tpssh"),
            )
        )


def test_rohf_handles_a_single_unpaired_electron() -> None:
    """ROHF считает предельный случай открытой оболочки — атом водорода.

    Прежняя версия этого теста утверждала, что ROHF не реализован и обязан
    отклоняться. Метод реализован, поэтому проверяется содержательное: при
    ``n_beta = 0`` ограничение ROHF вырождается, и энергия обязана совпасть с
    UHF — двух детерминантных степеней свободы, которые можно было бы
    ограничить, просто нет.
    """
    hydrogen = Molecule(
        name="h", atoms=(Atom(symbol="H", position=(0.0, 0.0, 0.0)),), multiplicity=2
    )

    def energy(spin: SpinTreatment) -> float:
        outcome = _run(
            molecule=hydrogen,
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=spin),
            ),
        )
        assert outcome.converged
        assert outcome.spin_squared == pytest.approx(0.75, abs=1e-10)
        return float(outcome.energy_hartree)

    assert energy(SpinTreatment.ROHF) == pytest.approx(energy(SpinTreatment.UHF), abs=1e-10)


def test_uhf_computes_open_shell_single_point() -> None:
    """Атом водорода: UHF считает открытую оболочку и сообщает <S^2>."""
    hydrogen = Molecule(
        name="h", atoms=(Atom(symbol="H", position=(0.0, 0.0, 0.0)),), multiplicity=2
    )
    result = _run(
        molecule=hydrogen,
        spec=CalculationSpec(
            task=Task.SINGLE_POINT,
            method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.UHF),
        ),
    )
    assert result.converged
    # Дублет: <S^2> = S(S+1) = 0.75 без загрязнения.
    assert result.spin_squared is not None
    assert abs(result.spin_squared - 0.75) < 1e-8
    # Канал β пуст, поэтому его границы не сообщаются — выдумывать их нельзя.
    assert result.beta_homo_energy_hartree is None
    assert result.beta_lumo_energy_hartree is not None


def test_uhf_optimization_of_an_open_shell_converges() -> None:
    """Оптимизация открытой оболочки доходит до конца и меняет геометрию.

    Раньше такой запрос отклонялся через ``MethodNotAvailableError``
    (``uhf-optimization``): энергии UHF были, а аналитических градиентов не
    было. Теперь они есть, и продолжать отклонять означало бы занижать реальные
    возможности (§54 ТЗ).
    """
    ch = Molecule.from_xyz(
        (Path(__file__).parent / "fixtures" / "ch-radical.xyz").read_text(encoding="utf-8"),
        name="ch",
        multiplicity=2,
    )
    result = _run(
        molecule=ch,
        spec=CalculationSpec(
            task=Task.OPTIMIZATION,
            method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.UHF),
            # По умолчанию спецификация просит избыточные внутренние координаты,
            # а они не реализованы: без явного указания запрос отклонился бы
            # совсем не по той причине, которую проверяет тест.
            optimization=OptimizationSpec(coordinates="cartesian"),
        ),
    )
    assert result.converged
    assert result.final_molecule is not None
    assert result.optimization_steps is not None and result.optimization_steps > 0
    # Свойства открытой оболочки доходят до результата: <S^2> скрывать нельзя.
    assert result.spin_squared is not None


def test_uhf_reproduces_rhf_on_closed_shell() -> None:
    """Для замкнутой оболочки UHF обязан дать ровно RHF-решение."""
    water = Molecule.from_xyz(
        (Path(__file__).parent / "fixtures" / "water.xyz").read_text(encoding="utf-8"),
        name="water",
    )
    rhf = MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.RHF)
    uhf = MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.UHF)
    restricted = _run(molecule=water, spec=CalculationSpec(task=Task.SINGLE_POINT, method=rhf))
    unrestricted = _run(molecule=water, spec=CalculationSpec(task=Task.SINGLE_POINT, method=uhf))
    assert unrestricted.energy_hartree == pytest.approx(restricted.energy_hartree, abs=1e-12)
    assert unrestricted.spin_squared == pytest.approx(0.0, abs=1e-10)


def test_uhf_optimization_is_not_rejected_any_more() -> None:
    """Аналитические градиенты UHF реализованы — оптимизация открытой оболочки идёт.

    Прежняя версия этого теста утверждала обратное и проходила лишь потому, что
    спецификация по умолчанию просила нереализованную систему координат. Теперь
    проверяется содержательное: расчёт доходит до результата, а ⟨S²⟩ в нём есть.
    """
    hydrogen = Molecule(
        name="h2",
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, 1.4)),
        ),
        multiplicity=2,
        charge=1,
    )
    result = _run(
        molecule=hydrogen,
        spec=CalculationSpec(
            task=Task.OPTIMIZATION,
            method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.UHF),
            optimization=OptimizationSpec(coordinates="cartesian", max_steps=5),
        ),
    )
    assert result.optimization_steps is not None
    assert result.spin_squared is not None


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


def _h2(distance: float) -> Molecule:
    return Molecule(
        name="h2",
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, distance)),
        ),
    )


def _optimization_spec(
    *, max_steps: int = 30, frozen_atoms: tuple[int, ...] = ()
) -> CalculationSpec:
    """Спецификация оптимизации в единственной реализованной системе координат."""
    return CalculationSpec(
        task=Task.OPTIMIZATION,
        method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
        optimization=OptimizationSpec(
            coordinates="cartesian", max_steps=max_steps, frozen_atoms=frozen_atoms
        ),
    )


def test_optimization_returns_the_geometry_its_numbers_belong_to() -> None:
    """Числа в результате относятся к возвращённой геометрии, а не к исходной.

    Это главное требование к оптимизации: энергия из последней итерации
    оптимизатора, приписанная другой структуре, выглядит правдоподобно и
    делает результат непригодным.
    """
    engine = ReferenceEngine()
    result = engine.run(
        EngineRequest(job_id="job-opt", molecule=_h2(0.95), spec=_optimization_spec())
    )
    assert result.converged
    assert result.final_molecule is not None

    first = np.array(result.final_molecule.atoms[0].position)
    second = np.array(result.final_molecule.atoms[1].position)
    assert float(np.linalg.norm(second - first)) == pytest.approx(0.71223, abs=2e-3)

    # Независимая перепроверка: SCF на возвращённой геометрии даёт ту же энергию.
    basis = build_basis("sto-3g", result.final_molecule)
    assert run_rhf(basis, result.final_molecule, ScfSettings()).total_energy == pytest.approx(
        result.energy_hartree, rel=1e-10
    )


def test_optimization_lowers_the_energy() -> None:
    """Оптимизация из растянутой геометрии понижает энергию."""
    engine = ReferenceEngine()
    start_energy = engine.run(
        EngineRequest(
            job_id="job-sp",
            molecule=_h2(0.95),
            spec=CalculationSpec(
                task=Task.SINGLE_POINT,
                method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
            ),
        )
    ).energy_hartree
    result = engine.run(
        EngineRequest(job_id="job-opt", molecule=_h2(0.95), spec=_optimization_spec())
    )
    assert result.energy_hartree < start_energy


def test_optimization_reports_its_own_quality_check() -> None:
    """Сходимость оптимизации — отдельная проверка качества."""
    engine = ReferenceEngine()
    result = engine.run(
        EngineRequest(job_id="job-opt", molecule=_h2(0.95), spec=_optimization_spec())
    )
    checks = result.checks_by_name()
    assert checks["optimization_converged"].verdict is QualityVerdict.PASS
    assert checks["energy_decomposition"].verdict is QualityVerdict.PASS
    assert checks["fock_density_commutator"].verdict is QualityVerdict.PASS
    assert [record.stage for record in result.timings] == [
        "optimization",
        "final-scf",
        "properties",
    ]


def test_exhausted_optimization_is_reported_not_hidden() -> None:
    """Не сошлось за отведённое число шагов — результат с предупреждением.

    Статус ``converged`` обязан это отражать: молча вернуть несошедшуюся
    геометрию как равновесную — ровно тот обман, который запрещён §54 ТЗ.
    """
    engine = ReferenceEngine()
    result = engine.run(
        EngineRequest(job_id="job-opt", molecule=_h2(0.95), spec=_optimization_spec(max_steps=1))
    )
    assert not result.converged
    assert result.final_molecule is not None
    assert any(warning.key == "warning.optimization_not_converged" for warning in result.warnings)
    assert result.checks_by_name()["optimization_converged"].verdict is QualityVerdict.FAIL


def test_frozen_atom_survives_the_engine_round_trip() -> None:
    """Замороженный атом остаётся на месте и после прохода через движок."""
    engine = ReferenceEngine()
    start = _h2(0.95)
    result = engine.run(
        EngineRequest(
            job_id="job-opt",
            molecule=start,
            spec=_optimization_spec(frozen_atoms=(0,)),
        )
    )
    assert result.final_molecule is not None
    assert result.final_molecule.atoms[0].position == start.atoms[0].position


def test_redundant_internal_coordinates_are_rejected_honestly() -> None:
    """Дефолт спецификации — избыточные внутренние координаты, которых нет.

    Подменять их декартовыми молча нельзя: это другая система координат и
    другая скорость сходимости, пользователь должен знать, что именно считается.
    """
    engine = ReferenceEngine()
    spec = CalculationSpec(
        task=Task.OPTIMIZATION,
        method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
        optimization=OptimizationSpec(coordinates="redundant_internal"),
    )
    with pytest.raises(MethodNotAvailableError):
        engine.run(EngineRequest(job_id="job-opt", molecule=_h2(0.95), spec=spec))


def test_fingerprint_distinguishes_initial_and_final_structure() -> None:
    """Отпечаток учитывает итоговую геометрию, иначе две оптимизации сливаются."""
    engine = ReferenceEngine()
    result = engine.run(
        EngineRequest(job_id="job-opt", molecule=_h2(0.95), spec=_optimization_spec())
    )
    components = result.fingerprint.components
    assert components["final_structure"]
    assert components["final_structure"] != components["initial_structure"]


def test_assert_supported_returns_the_basis_it_validated() -> None:
    """Предпроверка возвращает базис — её используют CLI, GUI и REST."""
    engine = ReferenceEngine()
    assert engine.assert_supported(_spec("sto-3g")) == "sto-3g"
    with pytest.raises(MethodNotAvailableError):
        engine.assert_supported(CalculationSpec(task=Task.IRC))


def _option_spec(**overrides: object) -> CalculationSpec:
    """Спецификация RHF/STO-3G с переопределёнными полями — для проверок отказов."""
    base: dict[str, object] = {
        "task": Task.SINGLE_POINT,
        "method": MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
    }
    base.update(overrides)
    return CalculationSpec(**base)  # type: ignore[arg-type]


def test_unimplemented_scf_strategies_are_rejected_not_silently_skipped() -> None:
    """Запрос EDIIS/SOSCF отклоняется, а не выполняется без них.

    Это ключевой случай §54 ТЗ: расчёт сошёлся бы и вернул число, посчитанное
    другим алгоритмом. Пользователь выбрал метод — он обязан получить либо его,
    либо отказ.
    """
    engine = ReferenceEngine()
    for strategies in (("ediis",), ("diis", "soscf"), ("ediis", "damping", "level_shift")):
        with pytest.raises(MethodNotAvailableError):
            engine.assert_supported(_option_spec(scf=ScfSpec(fallback_strategies=strategies)))
    with pytest.raises(MethodNotAvailableError):
        engine.assert_supported(_option_spec(scf=ScfSpec(stability_analysis=True)))
    with pytest.raises(MethodNotAvailableError):
        engine.assert_supported(_option_spec(scf=ScfSpec(fractional_occupations=True)))
    # Реализованные стратегии проходят.
    assert engine.assert_supported(
        _option_spec(scf=ScfSpec(fallback_strategies=("diis", "damping")))
    )


def test_unimplemented_optimizer_options_are_rejected() -> None:
    """Ограничения координат и обновление Бофилла отклоняются явно.

    Движок всегда применяет BFGS и не читает ``constraints``. Молча
    проигнорированное ограничение вернуло бы геометрию, которую пользователь
    счёл бы удовлетворяющей условию, — худший вид неправды, потому что
    результат выглядит успешным.
    """
    engine = ReferenceEngine()
    constrained = _option_spec(
        task=Task.OPTIMIZATION,
        optimization=OptimizationSpec(
            constraints=(CoordinateConstraint(atoms=(0, 1), value=1.0),),
        ),
    )
    with pytest.raises(MethodNotAvailableError):
        engine.assert_supported(constrained)
    for update in ("bofill", "none"):
        with pytest.raises(MethodNotAvailableError):
            engine.assert_supported(
                _option_spec(
                    task=Task.OPTIMIZATION,
                    optimization=OptimizationSpec(hessian_update=update),
                )
            )
    assert engine.assert_supported(
        _option_spec(task=Task.OPTIMIZATION, optimization=OptimizationSpec(hessian_update="bfgs"))
    )


def test_default_specification_is_executable() -> None:
    """Спецификация по умолчанию обязана проходить валидацию.

    Прежний дефолт ``coordinates="redundant_internal"`` описывал схему, которой
    в движке нет, поэтому любой запрос с настройками по умолчанию отклонялся.
    Дефолт, требующий правки перед использованием, — это ловушка, а не
    convenience.
    """
    optimization = OptimizationSpec()
    assert optimization.coordinates == "cartesian"
    assert ReferenceEngine().assert_supported(_option_spec(task=Task.OPTIMIZATION)) == "sto-3g"
    # И дефолтный список стратегий SCF состоит только из реализованных.
    for strategy in ScfSpec().fallback_strategies:
        assert default_registry().get(f"scf:{strategy}").availability.is_usable, strategy


def test_fallback_strategies_actually_whitelist_the_strategies() -> None:
    """Поле — белый список, а не пожелание: исключённая стратегия не применяется.

    Проверяется на отображении в настройки решателя: убрать стратегию из списка
    и получить расчёт с ней было бы той же неправдой, что и обещать
    нереализованное.
    """
    full = _scf_settings(
        _option_spec(
            scf=ScfSpec(
                fallback_strategies=("diis", "damping", "level_shift"),
                damping=0.5,
                level_shift=0.25,
            )
        )
    )
    assert full.damping_rounds > 0
    assert full.diis_start <= full.max_iterations

    without = _scf_settings(
        _option_spec(scf=ScfSpec(fallback_strategies=("diis",), damping=0.5, level_shift=0.25))
    )
    assert without.damping_rounds == 0
    assert without.diis_start <= without.max_iterations

    no_diis = _scf_settings(_option_spec(scf=ScfSpec(fallback_strategies=("damping",))))
    assert no_diis.diis_start > no_diis.max_iterations, "DIIS должен быть выключен"


def test_rohf_single_point_reports_exact_spin() -> None:
    """ROHF доходит до результата через движок и несёт ⟨S²⟩ = S(S+1).

    Проверка сквозного пути: реестр пропускает ``spin:rohf``, движок выбирает
    ROHF-ветку, а в результате видно главное отличие метода от UHF — отсутствие
    спинового загрязнения.
    """
    radical = Molecule.from_xyz(
        (Path(__file__).parent / "fixtures" / "ch-radical.xyz").read_text(encoding="utf-8"),
        multiplicity=2,
    )
    result = _run(
        molecule=radical,
        spec=CalculationSpec(
            task=Task.SINGLE_POINT,
            method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g", spin=SpinTreatment.ROHF),
        ),
    )
    assert result.converged
    assert result.spin_squared == pytest.approx(0.75, abs=1e-10)
