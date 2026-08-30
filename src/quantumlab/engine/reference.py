"""Референсное расчётное ядро: единственная точка входа для Job Manager'а.

Реализует протокол :class:`~quantumlab.engine.contracts.QuantumEngine` поверх
NumPy-реализаций базисов, интегралов и RHF. Назначение — быть **эталоном
корректности** (ADR-002): любые будущие backend'ы (C++/OpenMP, CUDA, внешние
пакеты) обязаны совпадать с ним в пределах заявленной точности.

Что ядро умеет и чего не умеет
------------------------------
Умеет: ``single_point`` методом RHF в любом из зарегистрированных базисов.

Не умеет — и сообщает об этом штатной ошибкой до начала вычислений, а не
выдаёт правдоподобное число:

* UHF/ROHF: только замкнутые оболочки, нечётное число электронов отклоняется;
* DFT, MP2, CC: отсутствуют как код;
* градиенты, оптимизация, частоты, сканирования;
* сферическая схема базисов: наборы, опубликованные с d/f в чистых угловых
  моментах, считаются в декартовой схеме (6 компонент вместо 5 для d). Это
  больший базис, энергии отличаются от табличных на ~1e-4 Eh; факт отражается
  в проверке качества ``basis_angular_scheme`` со статусом ``warn``.
"""

from __future__ import annotations

import os
import platform
import socket
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantumlab.domain.fingerprint import build_fingerprint
from quantumlab.domain.molecule import Molecule
from quantumlab.domain.result import (
    CalculationResult,
    CalculationWarning,
    EnvironmentInfo,
    OrbitalInfo,
    QualityCheck,
    QualityVerdict,
    TimingRecord,
)
from quantumlab.domain.spec import (
    CalculationSpec,
    OptimizationSpec,
    ScfSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine import integrals
from quantumlab.engine.basis import BasisSet, basis_angular_scheme, build_basis
from quantumlab.engine.checkpoint import (
    CheckpointError,
    assert_matches_job,
    read_scf_checkpoint,
    write_scf_checkpoint,
)
from quantumlab.engine.contracts import EngineRequest, ProgressReporter
from quantumlab.engine.dft import RksResult, run_rks
from quantumlab.engine.functional import (
    density_at_points,
    evaluate_basis,
    get_functional,
)
from quantumlab.engine.gradients import rhf_gradient, rks_gradient, uhf_gradient
from quantumlab.engine.optimizer import OptimizationSettings, optimize_geometry
from quantumlab.engine.quadrature import QuadratureGrid, build_grid
from quantumlab.engine.registry import CapabilityRegistry, default_registry
from quantumlab.engine.scf import (
    PrecomputedIntegrals,
    RohfResult,
    ScfSettings,
    UhfResult,
    build_fock,
    build_integrals,
    canonical_orthogonalizer,
    coulomb_matrix,
    exchange_matrix,
    run_rhf,
    run_rohf,
    run_uhf,
    spin_population,
)
from quantumlab.engine.scf import ScfResult as RhfResult
from quantumlab.engine.vibrations import numerical_hessian, vibrational_analysis
from quantumlab.errors import (
    CombinationUnavailableError,
    JobCheckpointInvalidError,
    ScfNotConvergedError,
)
from quantumlab.version import __version__

#: Имя ядра — попадает в отпечаток расчёта и в журнал.
ENGINE_NAME = "quantumlab-reference"

#: Backend, который фактически используется: один поток, dense float64.
ENGINE_BACKEND = "numpy-dense-cpu"

#: Точность, с которой tr(D·S) должен равняться числу электронов.
_ELECTRON_COUNT_TOLERANCE = 1e-6

#: Точность выполнения E = T + V_яд-эл + V_эл-эл + V_яд-яд на сошедшейся плотности.
_ENERGY_DECOMPOSITION_TOLERANCE = 1e-8

#: Точность условия стационарности ``FDS = SDF`` (следствие уравнений Рутана).
_FOCK_COMMUTATOR_TOLERANCE = 1e-6

#: Точность выполнения D′² = 2D′ в ортогональном базисе.
_IDEMPOTENCY_TOLERANCE = 1e-6

#: Насколько точно ∫ρ dV по квадратурной сетке обязано воспроизводить число
#: электронов. Строгий порог — для сеток, которыми мы готовы отчитываться;
#: между порогами выдаётся WARNING: грубая сетка действительно менее точна,
#: и скрывать это означало бы выдать число с неизвестной погрешностью.
_QUADRATURE_STRICT_TOLERANCE = 1e-6
_QUADRATURE_LOOSE_TOLERANCE = 1e-3

#: Избыток <S^2> над S(S+1), при котором сообщается о спиновом загрязнении.
#: Это порог **сообщения**, а не корректности: само значение <S^2> всегда
#: попадает в результат, поэтому пользователь видит величину в любом случае.
#: Умеренное загрязнение у радикалов физично (CH/STO-3G даёт избыток ~3e-3),
#: заметное — признак того, что однодетерминантное описание не годится.
_SPIN_CONTAMINATION_TOLERANCE = 0.05


def _total_memory_mb() -> int:
    """Полный объём памяти в МБ; ``0``, если определить не удалось.

    Возвращать выдуманное значение нельзя: отпечаток расчёта должен
    воспроизводиться, а ложные сведения об окружении этому мешают.
    """
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, IndexError, ValueError):
        return 0
    return 0


def _environment() -> EnvironmentInfo:
    """Собирает сведения об окружении для воспроизводимости (§40 ТЗ)."""
    return EnvironmentInfo(
        software_version=__version__,
        engine_version=__version__,
        engine_backend=ENGINE_BACKEND,
        python_version=platform.python_version(),
        os=f"{platform.system()} {platform.release()}",
        hostname=socket.gethostname(),
        cpu_model=platform.processor() or "unknown",
        cores=os.cpu_count() or 1,
        memory_mb=_total_memory_mb(),
        gpu=None,
        mpi_ranks=1,
    )


@dataclass(frozen=True)
class _ScfSummary:
    """Сводка SCF, общая для RHF и UHF.

    ``_result`` не должен знать, каким методом получена энергия: иначе каждая
    новая спиновая схема тянула бы правку сборки результата и отпечатка.
    """

    total_energy: float
    iterations: int
    converged: bool
    spin_squared: float | None = None
    beta_homo: float | None = None
    beta_lumo: float | None = None


@dataclass(frozen=True, slots=True)
class _Properties:
    """Производные величины, вычисляемые после сходимости SCF.

    Возвращаем не словарь, а типизированную структуру: ``HOMO`` может
    отсутствовать у системы без занятых орбиталей, и это различие должно быть
    видно в сигнатуре, а не всплывать как ошибка приведения типов.
    """

    orbitals: tuple[OrbitalInfo, ...]
    homo: float | None
    lumo: float | None
    gap: float | None
    dipole_debye: float
    beta_homo: float | None = None
    beta_lumo: float | None = None


CheckpointSink = Callable[[str], None]
"""Приёмник контрольной точки.

Движок отдаёт сериализованное состояние и не знает, куда оно попадёт: выбор
хранилища — забота Job Manager'а.
"""


class ReferenceEngine:
    """NumPy-реализация ядра: RHF в точке (single point)."""

    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        """Создаёт ядро.

        Реестр можно передать извне: так все интерфейсы (CLI, GUI, REST)
        проверяют доступность методов по одному и тому же набору возможностей,
        а плагины могут расширять его до запуска расчёта.
        """
        self._registry = registry or default_registry()

    @property
    def name(self) -> str:
        """Имя ядра."""
        return ENGINE_NAME

    @property
    def version(self) -> str:
        """Версия ядра — входит в отпечаток расчёта."""
        return __version__

    def supported_tasks(self) -> Sequence[str]:
        """Задачи, которые ядро действительно умеет выполнять."""
        return (Task.SINGLE_POINT.value, Task.OPTIMIZATION.value, Task.FREQUENCIES.value)

    # ------------------------------------------------------------------ #
    # Основной вход
    # ------------------------------------------------------------------ #
    def run(
        self,
        request: EngineRequest,
        *,
        progress: ProgressReporter | None = None,
        checkpoint_sink: CheckpointSink | None = None,
    ) -> CalculationResult:
        """Выполняет расчёт и возвращает результат с проверками качества.

        Любая неподдерживаемая комбинация (задача, метод, спин, базис, система
        координат оптимизации) отклоняется штатной ошибкой до начала вычислений
        — пользователь получает «этот метод пока недоступен», а не
        правдоподобное число (§54 ТЗ).
        """
        spec = request.spec
        basis_name = self.assert_supported(spec)
        if spec.task is Task.OPTIMIZATION:
            return self._run_optimization(request, basis_name, progress=progress)
        if spec.task is Task.FREQUENCIES:
            return self._run_frequencies(request, basis_name, progress=progress)
        return self._run_single_point(
            request, basis_name, progress=progress, checkpoint_sink=checkpoint_sink
        )

    def _restore_density(
        self, request: EngineRequest, basis_name: str, prepared: PrecomputedIntegrals
    ) -> np.ndarray | None:
        """Извлекает стартовую плотность из контрольной точки.

        Возвращает ``None``, если контрольной точки нет: это обычная ситуация,
        а не ошибка. А вот непригодная контрольная точка — ошибка штатная и
        локализуемая: молча считать с ней нельзя, потому что расчёт сошёлся бы
        и выдал число, относящееся к другой системе.
        """
        if request.checkpoint is None:
            return None
        try:
            state = read_scf_checkpoint(request.checkpoint, overlap=prepared.overlap)
            assert_matches_job(state, molecule=request.molecule, basis=basis_name)
        except CheckpointError as error:
            raise JobCheckpointInvalidError(str(error)) from error
        return state.density

    # ------------------------------------------------------------------ #
    # Одноточечный расчёт
    # ------------------------------------------------------------------ #
    def _run_single_point(
        self,
        request: EngineRequest,
        basis_name: str,
        *,
        progress: ProgressReporter | None,
        checkpoint_sink: CheckpointSink | None = None,
    ) -> CalculationResult:
        """Одноточечный расчёт в фиксированной геометрии.

        Контрольные точки поддерживаются только в ветке RHF. Для остальных
        ``checkpoint_sink`` не вызывается: это значит, что рестарт для них
        недоступен, и ``job resume`` честно откажет, а не начнёт расчёт заново,
        выдавая его за продолжение.
        """
        method = request.spec.method
        if method is not None and method.theory is TheoryFamily.DFT:
            return self._run_single_point_rks(request, basis_name, progress=progress)
        if method is not None and method.spin is SpinTreatment.UHF:
            return self._run_single_point_uhf(request, basis_name, progress=progress)
        if method is not None and method.spin is SpinTreatment.ROHF:
            return self._run_single_point_rohf(request, basis_name, progress=progress)
        return self._run_single_point_rhf(
            request, basis_name, progress=progress, checkpoint_sink=checkpoint_sink
        )

    def _run_single_point_rks(
        self, request: EngineRequest, basis_name: str, *, progress: ProgressReporter | None
    ) -> CalculationResult:
        """RKS (LDA) в фиксированной геометрии.

        Сетка и значения базисных функций строятся здесь, а не внутри решателя:
        проверки качества обязаны пользоваться **той же** сеткой, на которой
        считалась энергия, иначе сравнение шло бы между двумя разными численными
        схемами и его результат ничего бы не значил.
        """
        spec = request.spec
        timings: list[TimingRecord] = []
        method = spec.method
        # Функционал не выбран пользователем — подставляем единственный
        # реализованный, но **сообщаем об этом** в предупреждениях: молчаливый
        # выбор метода расчёта сделал бы результат невоспроизводимым для того,
        # кто читает только число (§8, §54 ТЗ).
        # ``MethodSpec`` не пропускает DFT без функционала, поэтому ветки
        # «подставить что-нибудь по умолчанию» здесь быть не должно: молчаливый
        # выбор метода сделал бы результат невоспроизводимым (§54 ТЗ). Проверка
        # ниже — защита инварианта, а не обработка штатного случая.
        if method is None or method.functional is None:
            msg = "DFT-расчёт требует явного обменно-корреляционного функционала."
            raise ValueError(msg)
        functional = get_functional(method.functional)

        started = time.perf_counter()
        basis = build_basis(basis_name, request.molecule)
        timings.append(_timing("basis", started))
        _report(progress, 5.0, "basis", functions=basis.n_functions)

        started = time.perf_counter()
        prepared = build_integrals(basis, request.molecule)
        dipole_integrals = integrals.build_dipole_integrals(basis, request.molecule)
        timings.append(_timing("integrals", started))
        _report(progress, 35.0, "integrals")

        started = time.perf_counter()
        grid = build_grid(request.molecule, spec.grid.preset)
        basis_values = evaluate_basis(basis, request.molecule, grid.points)
        timings.append(_timing("grid", started))
        _report(progress, 45.0, "grid", points=grid.n_points)

        started = time.perf_counter()
        rks = run_rks(
            basis,
            request.molecule,
            functional,
            _scf_settings(spec),
            integrals=prepared,
            grid=grid,
        )
        timings.append(_timing("scf", started))
        _report(progress, 85.0, "scf", iterations=rks.iterations, converged=rks.converged)

        started = time.perf_counter()
        properties = _properties(rks.as_scf_result(), request.molecule, dipole_integrals)
        checks = _quality_checks_rks(rks, basis, request.molecule, prepared, grid, basis_values)
        timings.append(_timing("properties", started))
        _report(progress, 100.0, "properties")

        return self._result(
            request,
            molecule=request.molecule,
            basis=basis,
            scf=_ScfSummary(
                total_energy=rks.total_energy,
                iterations=rks.iterations,
                converged=rks.converged,
            ),
            properties=properties,
            checks=checks,
            timings=timings,
            warnings=_warnings_rks(
                rks, basis, request.molecule, grid, pruning_requested=spec.grid.prune
            ),
            final_molecule=None,
        )

    def _run_single_point_rhf(
        self,
        request: EngineRequest,
        basis_name: str,
        *,
        progress: ProgressReporter | None,
        checkpoint_sink: CheckpointSink | None = None,
    ) -> CalculationResult:
        """RHF в фиксированной геометрии."""
        spec = request.spec
        timings: list[TimingRecord] = []

        started = time.perf_counter()
        basis = build_basis(basis_name, request.molecule)
        timings.append(_timing("basis", started))
        _report(progress, 5.0, "basis", functions=basis.n_functions)

        started = time.perf_counter()
        prepared = build_integrals(basis, request.molecule)
        dipole_integrals = integrals.build_dipole_integrals(basis, request.molecule)
        timings.append(_timing("integrals", started))
        _report(progress, 45.0, "integrals")

        started = time.perf_counter()
        rhf = run_rhf(
            basis,
            request.molecule,
            _scf_settings(spec),
            integrals=prepared,
            initial_density=self._restore_density(request, basis_name, prepared),
        )
        timings.append(_timing("scf", started))
        _report(progress, 85.0, "scf", iterations=rhf.iterations, converged=rhf.converged)
        if checkpoint_sink is not None:
            # Пишем состояние, даже если SCF не сошёлся: частично сошедшаяся
            # плотность — лучшее начальное приближение для продолжения, чем
            # догадка по остову.
            checkpoint_sink(
                write_scf_checkpoint(
                    molecule=request.molecule,
                    basis=basis_name,
                    density=rhf.density,
                    total_energy=rhf.total_energy,
                    iterations=rhf.iterations,
                )
            )

        started = time.perf_counter()
        properties = _properties(rhf, request.molecule, dipole_integrals)
        checks = _quality_checks(rhf, basis, request.molecule, prepared)
        timings.append(_timing("properties", started))
        _report(progress, 100.0, "properties")

        return self._result(
            request,
            molecule=request.molecule,
            basis=basis,
            scf=_ScfSummary(
                total_energy=rhf.total_energy,
                iterations=rhf.iterations,
                converged=rhf.converged,
            ),
            properties=properties,
            checks=checks,
            timings=timings,
            warnings=_warnings(rhf, basis, request.molecule),
            final_molecule=None,
        )

    def _run_single_point_uhf(
        self, request: EngineRequest, basis_name: str, *, progress: ProgressReporter | None
    ) -> CalculationResult:
        """UHF в фиксированной геометрии.

        Интегралы ровно те же, что и в RHF: UHF отличается только построением
        фокиана и наличием двух плотностей. Градиенты UHF не реализованы,
        поэтому оптимизация геометрии для открытой оболочки отклоняется выше.
        """
        spec = request.spec
        timings: list[TimingRecord] = []

        started = time.perf_counter()
        basis = build_basis(basis_name, request.molecule)
        timings.append(_timing("basis", started))
        _report(progress, 5.0, "basis", functions=basis.n_functions)

        started = time.perf_counter()
        prepared = build_integrals(basis, request.molecule)
        dipole_integrals = integrals.build_dipole_integrals(basis, request.molecule)
        timings.append(_timing("integrals", started))
        _report(progress, 45.0, "integrals")

        started = time.perf_counter()
        uhf = run_uhf(basis, request.molecule, _scf_settings(spec), integrals=prepared)
        timings.append(_timing("scf", started))
        _report(progress, 85.0, "scf", iterations=uhf.iterations, converged=uhf.converged)

        started = time.perf_counter()
        properties = _properties_uhf(uhf, request.molecule, dipole_integrals)
        checks = _quality_checks_uhf(uhf, basis, request.molecule, prepared)
        timings.append(_timing("properties", started))
        _report(progress, 100.0, "properties")

        return self._result(
            request,
            molecule=request.molecule,
            basis=basis,
            scf=_ScfSummary(
                total_energy=uhf.total_energy,
                iterations=uhf.iterations,
                converged=uhf.converged,
                spin_squared=uhf.s_squared,
                beta_homo=properties.beta_homo,
                beta_lumo=properties.beta_lumo,
            ),
            properties=properties,
            checks=checks,
            timings=timings,
            warnings=_warnings_uhf(uhf, basis, request.molecule),
            final_molecule=None,
        )

    def _run_single_point_rohf(
        self, request: EngineRequest, basis_name: str, *, progress: ProgressReporter | None
    ) -> CalculationResult:
        """ROHF в фиксированной геометрии.

        Интегралы и выражение для энергии те же, что и в UHF: различие только в
        допустимых плотностях — орбитали обоих каналов общие. Проверки качества
        и предупреждения переиспользуются от UHF, потому что физика та же;
        расходиться они не должны.

        Градиентов ROHF нет, поэтому оптимизация и частоты для него отклоняются
        на уровне реестра, а не считаются по силам другого метода.
        """
        spec = request.spec
        timings: list[TimingRecord] = []

        started = time.perf_counter()
        basis = build_basis(basis_name, request.molecule)
        timings.append(_timing("basis", started))
        _report(progress, 5.0, "basis", functions=basis.n_functions)

        started = time.perf_counter()
        prepared = build_integrals(basis, request.molecule)
        dipole_integrals = integrals.build_dipole_integrals(basis, request.molecule)
        timings.append(_timing("integrals", started))
        _report(progress, 45.0, "integrals")

        started = time.perf_counter()
        rohf = run_rohf(basis, request.molecule, _scf_settings(spec), integrals=prepared)
        timings.append(_timing("scf", started))
        _report(progress, 85.0, "scf", iterations=rohf.iterations, converged=rohf.converged)

        started = time.perf_counter()
        properties = _properties_uhf(rohf, request.molecule, dipole_integrals)
        checks = _quality_checks_uhf(rohf, basis, request.molecule, prepared)
        timings.append(_timing("properties", started))
        _report(progress, 100.0, "properties")

        return self._result(
            request,
            molecule=request.molecule,
            basis=basis,
            scf=_ScfSummary(
                total_energy=rohf.total_energy,
                iterations=rohf.iterations,
                converged=rohf.converged,
                spin_squared=rohf.s_squared,
                beta_homo=properties.beta_homo,
                beta_lumo=properties.beta_lumo,
            ),
            properties=properties,
            checks=checks,
            timings=timings,
            warnings=_warnings_uhf(rohf, basis, request.molecule),
            final_molecule=None,
        )

    # ------------------------------------------------------------------ #
    # Оптимизация геометрии
    # ------------------------------------------------------------------ #
    def _run_frequencies(
        self, request: EngineRequest, basis_name: str, *, progress: ProgressReporter | None
    ) -> CalculationResult:
        """Колебательные частоты: численный гессиан из аналитических градиентов.

        Расчёт в одной точке выполняется целиком (со своими проверками качества),
        а к нему добавляются частоты: дублировать SCF, свойства и проверки ради
        одной задачи означало бы держать два расходящихся мнения о том же
        расчёте.

        Частоты имеют смысл только в стационарной точке. Если сила превышает
        порог, расчёт **не отбрасывается и не подменяется оптимизацией** —
        пользователь попросил частоты, а не геометрию; вместо этого результат
        несёт явное предупреждение. То же самое с мнимыми частотами: одна мнимая
        частота означает седловую точку, и сообщить об этом обязан движок.
        """
        spec = request.spec
        base = self._run_single_point(request, basis_name, progress=progress)
        _report(progress, 15.0, "frequencies")

        started = time.perf_counter()
        hessian = numerical_hessian(
            request.molecule,
            lambda molecule: _solve_energy_and_gradient(spec, basis_name, molecule)[1],
        )
        vibrations = vibrational_analysis(hessian, request.molecule)
        _, gradient = _solve_energy_and_gradient(spec, basis_name, request.molecule)
        timings = [*base.timings, _timing("hessian", started)]

        warnings = list(base.warnings)
        max_force = float(np.max(np.abs(gradient)))
        if max_force > _STATIONARY_FORCE_TOLERANCE:
            warnings.append(
                CalculationWarning(
                    key="warning.frequencies_off_stationary",
                    params={
                        "max_force": f"{max_force:.3e}",
                        "threshold": f"{_STATIONARY_FORCE_TOLERANCE:.1e}",
                    },
                )
            )
        if vibrations.imaginary_frequencies:
            warnings.append(
                CalculationWarning(
                    key="warning.frequencies_imaginary",
                    params={
                        "values": ", ".join(
                            f"{value:.1f}" for value in vibrations.imaginary_frequencies
                        )
                    },
                )
            )
        _report(progress, 100.0, "frequencies", modes=len(vibrations.frequencies_cm1))

        return base.model_copy(
            update={
                "frequencies_cm1": vibrations.frequencies_cm1,
                "zero_point_energy_hartree": vibrations.zero_point_energy_hartree,
                "timings": tuple(timings),
                "warnings": tuple(warnings),
            }
        )

    def _run_optimization(
        self, request: EngineRequest, basis_name: str, *, progress: ProgressReporter | None
    ) -> CalculationResult:
        """Оптимизация геометрии: градиент → квазиньютоновские шаги → свойства.

        Числа в результате (энергия, орбитали, диполь) относятся к
        **оптимизированной** геометрии, а не к исходной. Поэтому в конце
        выполняется отдельный SCF на найденной структуре: брать энергию из
        последней итерации оптимизатора и приписывать её другой геометрии —
        ровно тот класс ошибок, который делает результат непригодным.
        """
        spec = request.spec
        options = spec.optimization
        budget = max(options.max_steps, 1)
        timings: list[TimingRecord] = []

        method = spec.method
        if method is not None and method.theory is TheoryFamily.DFT:
            if method.functional is None:
                msg = "DFT-расчёт требует явного обменно-корреляционного функционала."
                raise ValueError(msg)
            functional = get_functional(method.functional)
        else:
            functional = None
        is_uhf = method is not None and method.spin is SpinTreatment.UHF

        def energy_and_gradient(molecule: Molecule) -> tuple[float, np.ndarray]:
            total_energy, gradient = _solve_energy_and_gradient(spec, basis_name, molecule)
            done.append(1)
            _report(
                progress,
                min(5.0 + 80.0 * len(done) / (budget + 1), 85.0),
                "optimization",
                step=len(done),
            )
            return total_energy, gradient

        done: list[int] = []
        started = time.perf_counter()
        optimization = optimize_geometry(
            request.molecule, energy_and_gradient, _optimization_settings(spec)
        )
        timings.append(_timing("optimization", started))
        _report(progress, 90.0, "optimization", steps=optimization.steps)

        final = optimization.molecule
        started = time.perf_counter()
        basis = build_basis(basis_name, final)
        prepared = build_integrals(basis, final)
        dipole_integrals = integrals.build_dipole_integrals(basis, final)
        final_solution: RhfResult | RksResult | UhfResult
        if functional is not None:
            grid = build_grid(final, spec.grid.preset)
            basis_values = evaluate_basis(basis, final, grid.points)
            rks_final = run_rks(
                basis, final, functional, _scf_settings(spec), integrals=prepared, grid=grid
            )
            final_solution = rks_final
        elif is_uhf:
            uhf_final = run_uhf(basis, final, _scf_settings(spec), integrals=prepared)
            final_solution = uhf_final
        else:
            rhf_final = run_rhf(basis, final, _scf_settings(spec), integrals=prepared)
            final_solution = rhf_final
        timings.append(_timing("final-scf", started))

        started = time.perf_counter()
        # Свойства, проверки и предупреждения различаются по методу: у RKS к ним
        # добавляются квадратурные, у UHF — второй спиновой канал и <S^2>.
        # Вычисляются внутри своей ветви, где конкретный тип результата известен,
        # а не по общему объединению.
        if functional is not None:
            properties = _properties(rks_final, final, dipole_integrals)
            checks = _quality_checks_rks(
                rks_final, basis, final, prepared, grid, basis_values
            ) + _optimization_check(optimization)
            extra_warnings = _warnings_rks(
                rks_final, basis, final, grid, pruning_requested=spec.grid.prune
            )
        elif is_uhf:
            properties = _properties_uhf(uhf_final, final, dipole_integrals)
            checks = _quality_checks_uhf(uhf_final, basis, final, prepared) + _optimization_check(
                optimization
            )
            extra_warnings = _warnings_uhf(uhf_final, basis, final)
        else:
            properties = _properties(rhf_final, final, dipole_integrals)
            checks = _quality_checks(rhf_final, basis, final, prepared) + _optimization_check(
                optimization
            )
            extra_warnings = _warnings(rhf_final, basis, final)
        timings.append(_timing("properties", started))
        _report(progress, 100.0, "properties")

        return self._result(
            request,
            molecule=final,
            basis=basis,
            # У UHF на сводке лежат ещё <S^2> и границы канала beta: без них
            # оптимизация открытой оболочки вернула бы результат, из которого
            # молча исчезло спиновое загрязнение (§54 ТЗ).
            scf=_ScfSummary(
                total_energy=final_solution.total_energy,
                iterations=final_solution.iterations,
                converged=final_solution.converged,
                spin_squared=uhf_final.s_squared if is_uhf else None,
                beta_homo=properties.beta_homo,
                beta_lumo=properties.beta_lumo,
            ),
            properties=properties,
            checks=checks,
            timings=timings,
            warnings=extra_warnings + _optimization_warnings(optimization),
            final_molecule=final,
            initial_molecule=request.molecule,
            converged=final_solution.converged and optimization.converged,
            optimization_steps=optimization.steps,
        )

    # ------------------------------------------------------------------ #
    # Сборка результата
    # ------------------------------------------------------------------ #
    def _result(
        self,
        request: EngineRequest,
        *,
        molecule: Molecule,
        basis: BasisSet,
        scf: _ScfSummary,
        properties: _Properties,
        checks: tuple[QualityCheck, ...],
        timings: list[TimingRecord],
        warnings: tuple[CalculationWarning, ...],
        final_molecule: Molecule | None,
        initial_molecule: Molecule | None = None,
        converged: bool | None = None,
        optimization_steps: int | None = None,
        frequencies_cm1: tuple[float, ...] = (),
        zero_point_energy_hartree: float | None = None,
    ) -> CalculationResult:
        """Собирает ``CalculationResult``: отпечаток, проверки, окружение.

        ``initial_molecule`` задаётся только для расчётов, меняющих геометрию:
        отпечаток обязан различать исходную и конечную структуры, иначе две
        разные оптимизации из разных стартов выглядели бы одинаково.
        """
        spec = request.spec
        fingerprint = build_fingerprint(
            spec=spec,
            molecule=initial_molecule or molecule,
            software_version=__version__,
            engine_version=self.version,
            hardware={"backend": ENGINE_BACKEND, "threads": str(request.threads)},
            final_molecule=final_molecule,
        )
        del basis  # используется вызывающим кодом для проверок качества
        return CalculationResult(
            job_id=request.job_id,
            spec=spec,
            fingerprint=fingerprint,
            energy_hartree=scf.total_energy,
            scf_iterations=scf.iterations,
            converged=scf.converged if converged is None else converged,
            beta_homo_energy_hartree=scf.beta_homo,
            beta_lumo_energy_hartree=scf.beta_lumo,
            spin_squared=scf.spin_squared,
            homo_energy_hartree=properties.homo,
            lumo_energy_hartree=properties.lumo,
            gap_hartree=properties.gap,
            dipole_debye=properties.dipole_debye,
            orbitals=properties.orbitals,
            quality_checks=checks,
            timings=tuple(timings),
            warnings=warnings,
            environment=_environment(),
            final_molecule=final_molecule,
            optimization_steps=optimization_steps,
            frequencies_cm1=frequencies_cm1,
            zero_point_energy_hartree=zero_point_energy_hartree,
        )

    # ------------------------------------------------------------------ #
    # Проверки допустимости запроса
    # ------------------------------------------------------------------ #
    def _assert_scf_is_honoured(self, scf: ScfSpec) -> None:
        """Отклоняет параметры SCF, которых движок всё равно не выполнил бы.

        Без этой проверки запрос «EDIIS + SOSCF» дошёл бы до расчёта и вернулся
        успешным — с DIIS внутри. Пользователь получил бы число, посчитанное не
        тем методом, который он выбрал, и узнать об этом было бы неоткуда (§54 ТЗ).
        """
        for strategy in scf.fallback_strategies:
            self._registry.assert_available(f"scf:{strategy}")
        if scf.stability_analysis:
            self._registry.assert_available("scf:stability_analysis")
        if scf.fractional_occupations:
            self._registry.assert_available("scf:fractional_occupations")

    def _assert_optimization_is_honoured(self, optimization: OptimizationSpec) -> None:
        """То же для оптимизации: схема обновления гессиана и ограничения.

        Движок всегда применяет BFGS и не читает ``constraints``, поэтому оба
        параметра обязаны отклоняться явно: молча проигнорированное ограничение
        выдаёт геометрию, которую пользователь сочтёт удовлетворяющей условию.
        """
        self._registry.assert_available(f"optimizer:hessian_update:{optimization.hessian_update}")
        if optimization.constraints:
            self._registry.assert_available("optimizer:constraints")

    def _assert_spin_combination_is_honoured(self, spec: CalculationSpec) -> None:
        """Отклоняет сочетания, где спин реализован, а их комбинация с задачей — нет.

        Проверка доступности ``spin:uhf`` сама по себе недостаточна: UHF
        реализован для HF, но не для DFT. Без этой проверки запрос проходил
        валидацию, попадал в ветку RKS и падал необработанным ``ValueError``
        про нечётное число электронов — то есть пользователь получал аварийный
        сбой вместо честного «недоступно» (§54 ТЗ).

        То же с ROHF: реестр честно объявляет его пригодным только для энергии
        в одной точке, но объявление без читателя ничего не значит — движок
        проваливался в RHF и падал там. Проверка делает ограничение реальным.
        """
        method = spec.method
        if method is None:
            return

        if method.theory is TheoryFamily.DFT and method.spin is not SpinTreatment.RHF:
            # В ``combination`` — только технические идентификаторы: движок по
            # устройству не знает локали вызывающей стороны, а подставлять
            # русский текст в параметр значило бы зашить в код строку
            # интерфейса (§3 ТЗ).
            raise CombinationUnavailableError(
                f"DFT/{method.functional} + {method.spin.value}",
                "UKS (спиново-поляризованный DFT) не реализован: функционалы "
                "считаются только по полной плотности. Для открытой оболочки "
                "доступен метод HF с обработкой "
                f"{method.spin.value}.",
            )
        if method.spin is SpinTreatment.ROHF and spec.task is not Task.SINGLE_POINT:
            raise CombinationUnavailableError(
                f"ROHF + {spec.task.value}",
                "аналитических градиентов ROHF нет, поэтому нужны производные "
                "энергии. Для открытой оболочки доступны UHF (оптимизация и "
                "частоты) и ROHF (только энергия в одной точке).",
            )

    def assert_supported(self, spec: CalculationSpec) -> str:
        """Отклоняет запрос, который ядро не может выполнить корректно.

        Метод публичный намеренно: любой фронтенд — CLI, GUI, REST — обязан
        проверить спецификацию **до** создания задания, чтобы пользователь не
        увидел «идёт расчёт» для метода, которого не существует (§54 ТЗ).

        Возвращает имя базиса: оно нужно и для расчёта, и для сообщения
        об ошибке, поэтому извлекается здесь же.
        """
        implemented_tasks = (Task.SINGLE_POINT, Task.OPTIMIZATION, Task.FREQUENCIES)
        if spec.task not in implemented_tasks:
            self._registry.assert_available(f"task:{spec.task.value}")
        self._assert_scf_is_honoured(spec.scf)
        if spec.task is Task.OPTIMIZATION:
            coordinates = spec.optimization.coordinates
            self._registry.assert_available(f"coordinates:{coordinates}")
            self._assert_optimization_is_honoured(spec.optimization)
        method = spec.method
        if method is None:
            self._registry.assert_available("method:hf")
            msg = (
                "Для расчёта нужно явно указать метод и базис (spec.method). "
                "Базис по умолчанию не подставляется: молчаливый выбор базиса "
                "сделал бы результат невоспроизводимым."
            )
            raise ValueError(msg)
        if method.theory is not TheoryFamily.HF:
            self._registry.assert_available(f"method:{method.theory.value}")
        if method.functional is not None:
            self._registry.assert_available(f"functional:{method.functional}")
        # Дисперсионная поправка проверяется так же строго, как функционал:
        # без этой проверки план с D3-BJ выполнялся бы как расчёт без поправки,
        # то есть выдавал бы другое число под тем же описанием (§54 ТЗ).
        self._registry.assert_available(f"dispersion:{method.dispersion.value}")
        if method.spin is not SpinTreatment.RHF:
            self._registry.assert_available(f"spin:{method.spin.value}")
        self._assert_spin_combination_is_honoured(spec)
        self._registry.assert_available(f"basis:{method.basis}")
        return method.basis


def _timing(stage: str, started: float) -> TimingRecord:
    """Запись времени этапа. CPU-время не измеряется отдельно — один поток."""
    wall = time.perf_counter() - started
    return TimingRecord(stage=stage, wall_seconds=wall, cpu_seconds=wall)


#: Порог, ниже которого структуру можно считать стационарной точкой: частоты в
#: нестационарной точке лишены физического смысла, и об этом надо сказать.
#: Совпадает с порогом оптимизатора по силе, чтобы оба критерия не расходились.
_STATIONARY_FORCE_TOLERANCE: float = 4.5e-4


def _solve_energy_and_gradient(
    spec: CalculationSpec, basis_name: str, molecule: Molecule
) -> tuple[float, np.ndarray]:
    """Энергия и аналитический градиент в одной точке тем методом, что в спецификации.

    Используется и оптимизацией, и частотами: держать две копии выбора метода
    означало бы, что они разойдутся, и один путь начнёт считать другим методом.
    """
    method = spec.method
    basis = build_basis(basis_name, molecule)
    if method is not None and method.theory is TheoryFamily.DFT:
        if method.functional is None:
            msg = "DFT-расчёт требует явного обменно-корреляционного функционала."
            raise ValueError(msg)
        functional = get_functional(method.functional)
        # Сетка перестраивается на каждой геометрии, чтобы энергия оставалась той
        # же величиной, что и в расчёте в одной точке.
        grid = build_grid(molecule, spec.grid.preset)
        rks = run_rks(basis, molecule, functional, _scf_settings(spec), grid=grid)
        _require_converged(rks)
        return rks.total_energy, rks_gradient(basis, molecule, rks, grid, functional).gradient
    if method is not None and method.spin is SpinTreatment.UHF:
        uhf = run_uhf(basis, molecule, _scf_settings(spec))
        _require_converged(uhf)
        return uhf.total_energy, uhf_gradient(basis, molecule, uhf).gradient
    rhf = run_rhf(basis, molecule, _scf_settings(spec))
    _require_converged(rhf)
    return rhf.total_energy, rhf_gradient(basis, molecule, rhf).gradient


#: Все ключи предупреждений, которые может выдать движок.
#:
#: Тест сверяет этот список с каталогами переводов: предупреждение, у которого
#: нет английского варианта, в английском интерфейсе либо исчезнет, либо
#: останется русским — и то и другое нарушает §3 ТЗ.
WARNING_KEYS: tuple[str, ...] = (
    "warning.scf_not_converged",
    "warning.basis_spherical_scheme",
    "warning.dipole_origin_charged",
    "warning.grid_prune_unimplemented",
    "warning.grid_xc_integration",
    "warning.frequencies_off_stationary",
    "warning.frequencies_imaginary",
    "warning.optimization_not_converged",
)


def _require_converged(solution: RhfResult | RksResult | UhfResult) -> None:
    """Прерывает расчёт, если SCF не сошёлся.

    Градиент по несошедшейся плотности неверен, а оптимизация по неверному
    градиенту уходит в случайную точку — поэтому прерываем явно, а не
    возвращаем силы, в корректности которых нельзя поручиться.
    """
    if not solution.converged:
        raise ScfNotConvergedError(
            iterations=solution.iterations,
            residual=max(solution.history[-1].energy_change, 0.0) if solution.history else 0.0,
            attempts=solution.strategies_used,
        )


def _scf_settings(spec: CalculationSpec) -> ScfSettings:
    """Переносит параметры SCF из спецификации в настройки решателя."""
    scf = spec.scf
    # ``fallback_strategies`` — белый список, а не пожелание: стратегия, которой
    # в нём нет, не применяется. Иначе пользователь, убравший из списка сдвиг
    # уровней, получил бы расчёт со сдвигом и не узнал бы об этом.
    allowed = set(scf.fallback_strategies)
    use_damping = "damping" in allowed and scf.damping > 0
    use_level_shift = "level_shift" in allowed and scf.level_shift > 0
    return ScfSettings(
        max_iterations=scf.max_iterations,
        energy_tolerance=scf.energy_threshold,
        density_tolerance=scf.density_threshold,
        # DIIS нужен хотя бы из двух векторов — раньше экстраполировать нечего.
        # Выключается стартом позже последней итерации: отдельного флага у
        # решателя нет, а заводить второй способ отключить DIIS значило бы
        # держать два мнения об одном и том же.
        diis_start=max(scf.diis_start, 2) if "diis" in allowed else scf.max_iterations + 1,
        damping_factor=scf.damping if use_damping else 0.5,
        damping_rounds=2 if use_damping else 0,
        level_shift=scf.level_shift if use_level_shift else 0.25,
    )


def _optimization_settings(spec: CalculationSpec) -> OptimizationSettings:
    """Переносит параметры оптимизации из спецификации в настройки решателя."""
    options = spec.optimization
    return OptimizationSettings(
        max_steps=options.max_steps,
        max_force=options.max_force,
        rms_force=options.rms_force,
        max_displacement=options.max_displacement,
        rms_displacement=options.rms_displacement,
        trust_radius=options.trust_radius,
        frozen_atoms=options.frozen_atoms,
    )


def _optimization_check(optimization: object) -> tuple[QualityCheck, ...]:
    """Проверка качества по итогам оптимизации геометрии."""
    from quantumlab.engine.optimizer import OptimizationResult

    assert isinstance(optimization, OptimizationResult)
    forces = f"шагов: {optimization.steps}, max|F| = {optimization.max_force:.3e} э/бор"
    detail = forces if optimization.converged else f"{forces} — пороги спецификации не достигнуты"
    return (
        QualityCheck(
            name_key="optimization_converged",
            verdict=QualityVerdict.PASS if optimization.converged else QualityVerdict.FAIL,
            detail=detail,
        ),
    )


def _optimization_warnings(optimization: object) -> tuple[CalculationWarning, ...]:
    """Предупреждения оптимизации — пользователь обязан увидеть несходимость."""
    from quantumlab.engine.optimizer import OptimizationResult

    assert isinstance(optimization, OptimizationResult)
    if optimization.converged:
        return ()
    return (
        CalculationWarning(
            key="warning.optimization_not_converged",
            params={
                "steps": str(optimization.steps),
                "max_force": f"{optimization.max_force:.3e}",
            },
        ),
    )


def _report(progress: ProgressReporter | None, percent: float, stage: str, **extra: object) -> None:
    """Сообщает о прогрессе, если приёмник задан."""
    if progress is not None:
        progress.report(percent, stage, **extra)


def _properties(
    rhf: RhfResult | RksResult,
    molecule: Molecule,
    dipole_integrals: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> _Properties:
    """Орбитали, границы ЗМО/НСМО и дипольный момент в дебаях."""
    from quantumlab.engine.constants import AU_TO_DEBYE, angstrom_to_bohr

    n_occupied = molecule.n_electrons // 2
    orbitals = tuple(
        OrbitalInfo(
            index=index,
            energy_hartree=energy,
            occupation=2.0 if index < n_occupied else 0.0,
        )
        for index, energy in enumerate(rhf.orbital_energies)
    )
    homo = rhf.orbital_energies[n_occupied - 1] if n_occupied > 0 else None
    lumo = rhf.orbital_energies[n_occupied] if n_occupied < len(rhf.orbital_energies) else None
    gap = lumo - homo if homo is not None and lumo is not None else None

    nuclear = sum(
        atom.z * np.array([angstrom_to_bohr(value) for value in atom.position])
        for atom in molecule.atoms
    )
    electronic = np.array([float(np.sum(rhf.density * axis)) for axis in dipole_integrals])
    dipole_au = nuclear - electronic
    return _Properties(
        orbitals=orbitals,
        homo=homo,
        lumo=lumo,
        gap=gap,
        dipole_debye=float(np.linalg.norm(dipole_au) * AU_TO_DEBYE),
    )


def _properties_uhf(
    uhf: UhfResult | RohfResult,
    molecule: Molecule,
    dipole_integrals: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> _Properties:
    """Орбитали и диполь для открытой оболочки.

    В списке орбиталей — канал α: у UHF две независимые системы орбиталей, и
    смешивать их в один список означало бы выдать две разные величины под одним
    именем. Границы канала β возвращаются отдельными полями. Диполь считается
    по полной плотности (α + β) — это физически полная зарядовая плотность.
    """
    from quantumlab.engine.constants import AU_TO_DEBYE, angstrom_to_bohr

    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)
    orbitals = tuple(
        OrbitalInfo(
            index=index,
            energy_hartree=energy,
            occupation=1.0 if index < n_alpha else 0.0,
        )
        for index, energy in enumerate(uhf.alpha_energies)
    )
    homo = uhf.alpha_energies[n_alpha - 1] if n_alpha > 0 else None
    lumo = uhf.alpha_energies[n_alpha] if n_alpha < len(uhf.alpha_energies) else None
    beta_homo = uhf.beta_energies[n_beta - 1] if n_beta > 0 else None
    beta_lumo = uhf.beta_energies[n_beta] if n_beta < len(uhf.beta_energies) else None
    gap = lumo - homo if homo is not None and lumo is not None else None

    density_total = uhf.density_alpha + uhf.density_beta
    nuclear = sum(
        atom.z * np.array([angstrom_to_bohr(value) for value in atom.position])
        for atom in molecule.atoms
    )
    electronic = np.array([float(np.sum(density_total * axis)) for axis in dipole_integrals])
    return _Properties(
        orbitals=orbitals,
        homo=homo,
        lumo=lumo,
        beta_homo=beta_homo,
        beta_lumo=beta_lumo,
        gap=gap,
        dipole_debye=float(np.linalg.norm(nuclear - electronic) * AU_TO_DEBYE),
    )


def _quality_checks_uhf(
    uhf: UhfResult | RohfResult, basis: BasisSet, molecule: Molecule, prepared: PrecomputedIntegrals
) -> tuple[QualityCheck, ...]:
    """Проверки качества для UHF.

    RHF-проверки здесь неприменимы дословно: разложение энергии и условие
    стационарности у открытой оболочки строятся по каждому каналу отдельно, а
    полная плотность входит только в кулоновский член. Кроме того, добавлена
    проверка ⟨Ŝ²⟩ — у UHF возможно спиновое загрязнение, и оно должно быть
    видно в результате, а не оставаться внутри движка.
    """
    n_alpha, n_beta = spin_population(molecule.n_electrons, molecule.multiplicity)
    overlap = prepared.overlap
    repulsion = prepared.eri
    density_total = uhf.density_alpha + uhf.density_beta

    electron_count = float(np.trace(density_total @ overlap))

    kinetic = float(np.sum(density_total * integrals.build_kinetic(basis, molecule)))
    attraction = float(np.sum(density_total * integrals.build_nuclear_attraction(basis, molecule)))
    coulomb = float(np.einsum("uv,ls,uvls", density_total, density_total, repulsion))
    exchange_alpha = float(np.einsum("uv,ls,ulvs", uhf.density_alpha, uhf.density_alpha, repulsion))
    exchange_beta = float(np.einsum("uv,ls,ulvs", uhf.density_beta, uhf.density_beta, repulsion))
    # Обмен в UHF строится по плотности канала с занятием 1, поэтому перед
    # суммой обменов стоит ½, а не ¼ как в RHF. Проверка: для замкнутой
    # оболочки D^α = D^β = D_зан и D_полн = 2D_зан, тогда
    # ½(D^α·K^α + D^β·K^β) = ¼ D_полн·K(D_полн) — в точности множитель RHF.
    electron_electron = 0.5 * coulomb - 0.5 * (exchange_alpha + exchange_beta)
    decomposition_error = abs(
        kinetic + attraction + electron_electron + uhf.nuclear_repulsion - uhf.total_energy
    )

    commutator_error = 0.0
    idempotency_error = 0.0
    orthogonalizer = canonical_orthogonalizer(overlap)
    inverse = np.linalg.inv(orthogonalizer)
    for density, n_occupied in ((uhf.density_alpha, n_alpha), (uhf.density_beta, n_beta)):
        fock = (
            prepared.core
            + coulomb_matrix(density_total, repulsion)
            - exchange_matrix(density, repulsion)
        )
        commutator_error = max(
            commutator_error,
            float(np.max(np.abs(fock @ density @ overlap - overlap @ density @ fock))),
        )
        density_prime = inverse @ density @ inverse.T
        # Занятие канала равно 1, поэтому идемпотентность D'^2 = D' (без множителя 2).
        idempotency_error = max(
            idempotency_error,
            float(np.max(np.abs(density_prime @ density_prime - density_prime))),
        )
        del n_occupied  # число занятых нужно только для построения плотности

    s_exact = 0.5 * (n_alpha - n_beta) * (0.5 * (n_alpha - n_beta) + 1.0)
    scheme = basis_angular_scheme(basis.name)
    return (
        QualityCheck(
            name_key="scf_converged",
            verdict=QualityVerdict.PASS if uhf.converged else QualityVerdict.FAIL,
            detail=f"итераций: {uhf.iterations}, стратегии: {', '.join(uhf.strategies_used)}",
        ),
        QualityCheck(
            name_key="electron_count",
            verdict=(
                QualityVerdict.PASS
                if abs(electron_count - molecule.n_electrons) < _ELECTRON_COUNT_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"tr(D·S) = {electron_count:.8f}, ожидается {molecule.n_electrons}",
        ),
        QualityCheck(
            name_key="energy_decomposition",
            verdict=(
                QualityVerdict.PASS
                if decomposition_error < _ENERGY_DECOMPOSITION_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=(
                "E = T + V_яд-эл + V_эл-эл + V_яд-яд пересчитано по плотностям обоих каналов, "
                f"расхождение {decomposition_error:.3e} э"
            ),
        ),
        QualityCheck(
            name_key="fock_density_commutator",
            verdict=(
                QualityVerdict.PASS
                if commutator_error < _FOCK_COMMUTATOR_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"худший из каналов: max|FDS − SDF| = {commutator_error:.3e}",
        ),
        QualityCheck(
            name_key="density_idempotency",
            verdict=(
                QualityVerdict.PASS
                if idempotency_error < _IDEMPOTENCY_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"max|D'² − D'| = {idempotency_error:.3e} (занятие канала 1)",
        ),
        QualityCheck(
            name_key="spin_contamination",
            verdict=(
                QualityVerdict.PASS
                if uhf.s_squared <= s_exact + _SPIN_CONTAMINATION_TOLERANCE
                else QualityVerdict.WARNING
            ),
            detail=(
                f"<S^2> = {uhf.s_squared:.6f}, для чистого состояния {s_exact:.6f}; "
                f"избыток {uhf.s_squared - s_exact:+.6f}"
            ),
        ),
        QualityCheck(
            name_key="basis_angular_scheme",
            verdict=(QualityVerdict.PASS if scheme == "cartesian" else QualityVerdict.WARNING),
            detail=(
                "расчёт в декартовой схеме"
                if scheme == "cartesian"
                else f"базис {basis.name} опубликован в сферической схеме"
            ),
        ),
    )


def _warnings_uhf(
    uhf: UhfResult | RohfResult, basis: BasisSet, molecule: Molecule
) -> tuple[CalculationWarning, ...]:
    """Предупреждения UHF: несошедшийся SCF, схема базиса, начало отсчёта диполя."""
    warnings: list[CalculationWarning] = []
    if not uhf.converged:
        warnings.append(
            CalculationWarning(
                key="warning.scf_not_converged", params={"iterations": str(uhf.iterations)}
            )
        )
    warning = _angular_scheme_warning(basis)
    if warning:
        warnings.append(warning)
    dipole_warning = _dipole_origin_warning(molecule)
    if dipole_warning:
        warnings.append(dipole_warning)
    return tuple(warnings)


def _quality_checks(
    rhf: RhfResult | RksResult,
    basis: BasisSet,
    molecule: Molecule,
    prepared: PrecomputedIntegrals,
) -> tuple[QualityCheck, ...]:
    """Проверки, которые движок выполняет над собственным результатом (§26 ТЗ).

    Проверки нужны не для отчёта, а чтобы расхождение между математическими
    тождествами и фактическим результатом не прошло незамеченным.

    Тензор ERI берётся из ``prepared``, а не собирается заново: раньше проверка
    разложения энергии строила его второй раз за расчёт, и на воде/6-31G это
    стоило столько же, сколько весь SCF вместе с интегралами.
    """
    overlap = prepared.overlap
    electron_count = float(np.trace(rhf.density @ overlap))
    expected = molecule.n_electrons

    kinetic = float(np.sum(rhf.density * integrals.build_kinetic(basis, molecule)))
    attraction = float(np.sum(rhf.density * integrals.build_nuclear_attraction(basis, molecule)))
    repulsion = prepared.eri
    coulomb = float(np.einsum("uv,ls,uvls", rhf.density, rhf.density, repulsion))
    exchange = float(np.einsum("uv,ls,ulvs", rhf.density, rhf.density, repulsion))
    electron_electron = 0.5 * (coulomb - 0.5 * exchange)
    decomposition_error = abs(
        kinetic + attraction + electron_electron + rhf.nuclear_repulsion - rhf.total_energy
    )

    fock = build_fock(prepared.core, rhf.density, repulsion)
    commutator_error = float(
        np.max(np.abs(fock @ rhf.density @ overlap - overlap @ rhf.density @ fock))
    )

    orthogonalizer = canonical_orthogonalizer(overlap)
    inverse = np.linalg.inv(orthogonalizer)
    density_prime = inverse @ rhf.density @ inverse.T
    idempotency_error = float(np.max(np.abs(density_prime @ density_prime - 2.0 * density_prime)))

    scheme = basis_angular_scheme(basis.name)
    return (
        QualityCheck(
            name_key="scf_converged",
            verdict=QualityVerdict.PASS if rhf.converged else QualityVerdict.FAIL,
            detail=f"итераций: {rhf.iterations}, стратегии: {', '.join(rhf.strategies_used)}",
        ),
        QualityCheck(
            name_key="electron_count",
            verdict=(
                QualityVerdict.PASS
                if abs(electron_count - expected) < _ELECTRON_COUNT_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"tr(D·S) = {electron_count:.8f}, ожидается {expected}",
        ),
        QualityCheck(
            name_key="energy_decomposition",
            verdict=(
                QualityVerdict.PASS
                if decomposition_error < _ENERGY_DECOMPOSITION_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=(
                "E = T + V_яд-эл + V_эл-эл + V_яд-яд пересчитано по плотности, "
                f"расхождение {decomposition_error:.3e} э"
            ),
        ),
        QualityCheck(
            name_key="fock_density_commutator",
            verdict=(
                QualityVerdict.PASS
                if commutator_error < _FOCK_COMMUTATOR_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=(f"условие стационарности FDS = SDF, max|FDS − SDF| = {commutator_error:.3e}"),
        ),
        QualityCheck(
            name_key="density_idempotency",
            verdict=(
                QualityVerdict.PASS
                if idempotency_error < _IDEMPOTENCY_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"max|D′² − 2D′| = {idempotency_error:.3e}",
        ),
        QualityCheck(
            name_key="basis_angular_scheme",
            verdict=QualityVerdict.PASS if scheme == "cartesian" else QualityVerdict.WARNING,
            detail=(
                "базис опубликован в декартовой схеме — расчёт ей соответствует"
                if scheme == "cartesian"
                else (
                    "базис опубликован в сферической схеме, расчёт идёт в декартовой "
                    "(6 d-функций вместо 5); энергия ниже табличной примерно на 1e-4 Eh"
                )
            ),
        ),
    )


def _quality_checks_rks(
    rks: RksResult,
    basis: BasisSet,
    molecule: Molecule,
    prepared: PrecomputedIntegrals,
    grid: QuadratureGrid,
    basis_values: np.ndarray,
) -> tuple[QualityCheck, ...]:
    """Проверки качества для RKS.

    Отдельная функция, а не переиспользование RHF-проверок: разложение энергии
    у DFT другое (вместо обменного члена входит ``E_xc``), и подстановка
    HF-формул выдала бы FAIL на корректном результате — или, что хуже, PASS на
    неверном.

    Две проверки здесь ровно потому, что каждая ловит свой класс ошибки:
    ``quadrature_electron_count`` — неправильную меру интегрирования на сетке,
    ``energy_decomposition`` — подмену ``E_xc`` следом ``D·V_xc``.
    """
    overlap = prepared.overlap
    expected = molecule.n_electrons
    density = rks.density

    kinetic = float(np.sum(density * integrals.build_kinetic(basis, molecule)))
    attraction = float(np.sum(density * integrals.build_nuclear_attraction(basis, molecule)))
    coulomb = float(np.einsum("uv,ls,uvls", density, density, prepared.eri))
    # Гибрид добавляет долю точного обмена и в фокиан, и в разложение энергии.
    # Без обоих членов проверка выдала бы FAIL на корректном гибриде — или, что
    # хуже, PASS на гибриде, где точный обмен потерялся.
    alpha = rks.exact_exchange_fraction
    exchange_integral = (
        float(np.einsum("uv,ls,ulvs", density, density, prepared.eri)) if alpha > 0.0 else 0.0
    )

    rho = density_at_points(basis_values, density)
    grid_electrons = float(np.sum(grid.weights * rho))

    v_xc = rks.v_xc
    if v_xc is None:
        msg = "RKS-результат без обменно-корреляционного потенциала: проверки невозможны."
        raise ValueError(msg)
    fock = prepared.core + coulomb_matrix(density, prepared.eri) + v_xc
    if alpha > 0.0:
        fock = fock - 0.5 * alpha * exchange_matrix(density, prepared.eri)
    commutator_error = float(np.max(np.abs(fock @ density @ overlap - overlap @ density @ fock)))

    decomposition_error = abs(
        kinetic
        + attraction
        + 0.5 * coulomb
        + rks.xc_energy
        - 0.25 * alpha * exchange_integral
        + rks.nuclear_repulsion
        - rks.total_energy
    )

    orthogonalizer = canonical_orthogonalizer(overlap)
    inverse = np.linalg.inv(orthogonalizer)
    density_prime = inverse @ density @ inverse.T
    idempotency_error = float(np.max(np.abs(density_prime @ density_prime - 2.0 * density_prime)))

    grid_error = abs(grid_electrons - expected)
    if grid_error < _QUADRATURE_STRICT_TOLERANCE:
        grid_verdict = QualityVerdict.PASS
    elif grid_error < _QUADRATURE_LOOSE_TOLERANCE:
        grid_verdict = QualityVerdict.WARNING
    else:
        grid_verdict = QualityVerdict.FAIL

    scheme = basis_angular_scheme(basis.name)
    return (
        QualityCheck(
            name_key="scf_converged",
            verdict=QualityVerdict.PASS if rks.converged else QualityVerdict.FAIL,
            detail=f"итераций: {rks.iterations}, стратегии: {', '.join(rks.strategies_used)}",
        ),
        QualityCheck(
            name_key="electron_count",
            verdict=(
                QualityVerdict.PASS
                if abs(float(np.trace(density @ overlap)) - expected) < _ELECTRON_COUNT_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"tr(D·S) = {np.trace(density @ overlap):.8f}, ожидается {expected}",
        ),
        QualityCheck(
            name_key="quadrature_electron_count",
            verdict=grid_verdict,
            detail=(
                f"∫ρ dV по сетке из {rks.grid_points} точек = {grid_electrons:.8f}, "
                f"ожидается {expected}, расхождение {grid_error:.3e}"
            ),
        ),
        QualityCheck(
            name_key="energy_decomposition",
            verdict=(
                QualityVerdict.PASS
                if decomposition_error < _ENERGY_DECOMPOSITION_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=(
                "E = T + V_яд-эл + E_кулон + E_xc + V_яд-яд пересчитано по плотности, "
                f"E_xc = {rks.xc_energy:.8f} э, расхождение {decomposition_error:.3e} э"
            ),
        ),
        QualityCheck(
            name_key="fock_density_commutator",
            verdict=(
                QualityVerdict.PASS
                if commutator_error < _FOCK_COMMUTATOR_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"условие стационарности FDS = SDF, max|FDS − SDF| = {commutator_error:.3e}",
        ),
        QualityCheck(
            name_key="density_idempotency",
            verdict=(
                QualityVerdict.PASS
                if idempotency_error < _IDEMPOTENCY_TOLERANCE
                else QualityVerdict.FAIL
            ),
            detail=f"max|D′² − 2D′| = {idempotency_error:.3e}",
        ),
        QualityCheck(
            name_key="basis_angular_scheme",
            verdict=QualityVerdict.PASS if scheme == "cartesian" else QualityVerdict.WARNING,
            detail=(
                "базис опубликован в декартовой схеме — расчёт ей соответствует"
                if scheme == "cartesian"
                else (
                    "базис опубликован в сферической схеме, расчёт идёт в декартовой "
                    "(6 d-функций вместо 5); энергия ниже табличной примерно на 1e-4 Eh"
                )
            ),
        ),
    )


def _warnings_rks(
    rks: RksResult,
    basis: BasisSet,
    molecule: Molecule,
    grid: QuadratureGrid,
    *,
    pruning_requested: bool = False,
) -> tuple[CalculationWarning, ...]:
    """Предупреждения RKS: всё из RHF плюс качество сетки и границы метода."""
    warnings: list[CalculationWarning] = []
    if not rks.converged:
        warnings.append(
            CalculationWarning(
                key="warning.scf_not_converged", params={"iterations": str(rks.iterations)}
            )
        )
    scheme = _angular_scheme_warning(basis)
    if scheme:
        warnings.append(scheme)
    dipole = _dipole_origin_warning(molecule)
    if dipole:
        warnings.append(dipole)
    if pruning_requested:
        warnings.append(CalculationWarning(key="warning.grid_prune_unimplemented"))
    warnings.append(
        CalculationWarning(
            key="warning.grid_xc_integration",
            params={"points": str(grid.n_points), "preset": grid.preset.value},
        )
    )
    return tuple(warnings)


def _dipole_origin_warning(molecule: Molecule) -> CalculationWarning | None:
    """Предупреждение о начале отсчёта диполя у заряженной системы.

    Дипольный момент нейтральной системы от начала отсчёта не зависит, а у
    заряженной — зависит линейно. Мы считаем его от начала координат, поэтому
    для иона число воспроизводимо, но физический смысл имеет только вместе с
    указанием этой точки. Молча выдать его как «дипольный момент» нельзя.
    """
    if molecule.charge == 0:
        return None
    return CalculationWarning(
        key="warning.dipole_origin_charged", params={"charge": f"{molecule.charge:+d}"}
    )


def _angular_scheme_warning(basis: BasisSet) -> CalculationWarning | None:
    """Предупреждение о декартовой схеме, если базис опубликован в сферической.

    Общая для RHF и UHF: ключ один, и расхождение в формулировках означало бы,
    что пользователь видит разное предупреждение для одного и того же базиса.
    """
    if basis_angular_scheme(basis.name) == "cartesian":
        return None
    return CalculationWarning(key="warning.basis_spherical_scheme", params={"basis": basis.name})


def _warnings(
    rhf: RhfResult | RksResult, basis: BasisSet, molecule: Molecule
) -> tuple[CalculationWarning, ...]:
    """Предупреждения, которые обязан увидеть пользователь."""
    warnings: list[CalculationWarning] = []
    if not rhf.converged:
        warnings.append(
            CalculationWarning(
                key="warning.scf_not_converged", params={"iterations": str(rhf.iterations)}
            )
        )
    warning = _angular_scheme_warning(basis)
    if warning:
        warnings.append(warning)
    dipole_warning = _dipole_origin_warning(molecule)
    if dipole_warning:
        warnings.append(dipole_warning)
    return tuple(warnings)
