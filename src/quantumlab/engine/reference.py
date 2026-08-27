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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantumlab.domain.fingerprint import build_fingerprint
from quantumlab.domain.molecule import Molecule
from quantumlab.domain.result import (
    CalculationResult,
    EnvironmentInfo,
    OrbitalInfo,
    QualityCheck,
    QualityVerdict,
    TimingRecord,
)
from quantumlab.domain.spec import CalculationSpec, SpinTreatment, Task, TheoryFamily
from quantumlab.engine import integrals
from quantumlab.engine.basis import BasisSet, basis_angular_scheme, build_basis
from quantumlab.engine.contracts import EngineRequest, ProgressReporter
from quantumlab.engine.gradients import rhf_gradient
from quantumlab.engine.optimizer import OptimizationSettings, optimize_geometry
from quantumlab.engine.registry import CapabilityRegistry, default_registry
from quantumlab.engine.scf import (
    PrecomputedIntegrals,
    ScfSettings,
    UhfResult,
    build_fock,
    build_integrals,
    canonical_orthogonalizer,
    coulomb_matrix,
    exchange_matrix,
    run_rhf,
    run_uhf,
    spin_population,
)
from quantumlab.engine.scf import ScfResult as RhfResult
from quantumlab.errors import MethodNotAvailableError, ScfNotConvergedError
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
        return (Task.SINGLE_POINT.value, Task.OPTIMIZATION.value)

    # ------------------------------------------------------------------ #
    # Основной вход
    # ------------------------------------------------------------------ #
    def run(
        self, request: EngineRequest, *, progress: ProgressReporter | None = None
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
        return self._run_single_point(request, basis_name, progress=progress)

    # ------------------------------------------------------------------ #
    # Одноточечный расчёт
    # ------------------------------------------------------------------ #
    def _run_single_point(
        self, request: EngineRequest, basis_name: str, *, progress: ProgressReporter | None
    ) -> CalculationResult:
        """Одноточечный расчёт в фиксированной геометрии."""
        if request.spec.method is not None and request.spec.method.spin is SpinTreatment.UHF:
            return self._run_single_point_uhf(request, basis_name, progress=progress)
        return self._run_single_point_rhf(request, basis_name, progress=progress)

    def _run_single_point_rhf(
        self, request: EngineRequest, basis_name: str, *, progress: ProgressReporter | None
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
        rhf = run_rhf(basis, request.molecule, _scf_settings(spec), integrals=prepared)
        timings.append(_timing("scf", started))
        _report(progress, 85.0, "scf", iterations=rhf.iterations, converged=rhf.converged)

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

    # ------------------------------------------------------------------ #
    # Оптимизация геометрии
    # ------------------------------------------------------------------ #
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

        def energy_and_gradient(molecule: Molecule) -> tuple[float, np.ndarray]:
            basis = build_basis(basis_name, molecule)
            scf = run_rhf(basis, molecule, _scf_settings(spec))
            if not scf.converged:
                # Градиент по несошедшейся плотности неверен, а оптимизация по
                # неверному градиенту уходит в случайную точку. Прерываем явно.
                raise ScfNotConvergedError(
                    iterations=scf.iterations,
                    residual=max(scf.history[-1].energy_change, 0.0) if scf.history else 0.0,
                    attempts=scf.strategies_used,
                )
            gradient = rhf_gradient(basis, molecule, scf).gradient
            done.append(1)
            _report(
                progress,
                min(5.0 + 80.0 * len(done) / (budget + 1), 85.0),
                "optimization",
                step=len(done),
            )
            return scf.total_energy, gradient

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
        rhf = run_rhf(basis, final, _scf_settings(spec), integrals=prepared)
        timings.append(_timing("final-scf", started))

        started = time.perf_counter()
        properties = _properties(rhf, final, dipole_integrals)
        checks = _quality_checks(rhf, basis, final, prepared) + _optimization_check(optimization)
        timings.append(_timing("properties", started))
        _report(progress, 100.0, "properties")

        return self._result(
            request,
            molecule=final,
            basis=basis,
            scf=_ScfSummary(
                total_energy=rhf.total_energy,
                iterations=rhf.iterations,
                converged=rhf.converged,
            ),
            properties=properties,
            checks=checks,
            timings=timings,
            warnings=_warnings(rhf, basis, request.molecule) + _optimization_warnings(optimization),
            final_molecule=final,
            initial_molecule=request.molecule,
            converged=rhf.converged and optimization.converged,
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
        warnings: tuple[str, ...],
        final_molecule: Molecule | None,
        initial_molecule: Molecule | None = None,
        converged: bool | None = None,
        optimization_steps: int | None = None,
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
        )

    # ------------------------------------------------------------------ #
    # Проверки допустимости запроса
    # ------------------------------------------------------------------ #
    def assert_supported(self, spec: CalculationSpec) -> str:
        """Отклоняет запрос, который ядро не может выполнить корректно.

        Метод публичный намеренно: любой фронтенд — CLI, GUI, REST — обязан
        проверить спецификацию **до** создания задания, чтобы пользователь не
        увидел «идёт расчёт» для метода, которого не существует (§54 ТЗ).

        Возвращает имя базиса: оно нужно и для расчёта, и для сообщения
        об ошибке, поэтому извлекается здесь же.
        """
        if spec.task not in (Task.SINGLE_POINT, Task.OPTIMIZATION):
            self._registry.assert_available(f"task:{spec.task.value}")
        if spec.task is Task.OPTIMIZATION:
            coordinates = spec.optimization.coordinates
            self._registry.assert_available(f"coordinates:{coordinates}")
            if spec.method is not None and spec.method.spin is SpinTreatment.UHF:
                # UHF-энергия есть, а аналитических градиентов UHF нет. Считать
                # градиент по RHF-формулам для открытой оболочки означало бы
                # выдать неверные силы под видом результата (§54 ТЗ).
                raise MethodNotAvailableError("uhf-optimization")
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
        if method.spin is not SpinTreatment.RHF:
            self._registry.assert_available(f"spin:{method.spin.value}")
        self._registry.assert_available(f"basis:{method.basis}")
        return method.basis


def _timing(stage: str, started: float) -> TimingRecord:
    """Запись времени этапа. CPU-время не измеряется отдельно — один поток."""
    wall = time.perf_counter() - started
    return TimingRecord(stage=stage, wall_seconds=wall, cpu_seconds=wall)


def _scf_settings(spec: CalculationSpec) -> ScfSettings:
    """Переносит параметры SCF из спецификации в настройки решателя."""
    scf = spec.scf
    return ScfSettings(
        max_iterations=scf.max_iterations,
        energy_tolerance=scf.energy_threshold,
        density_tolerance=scf.density_threshold,
        # DIIS нужен хотя бы из двух векторов — раньше экстраполировать нечего.
        diis_start=max(scf.diis_start, 2),
        damping_factor=scf.damping if scf.damping > 0 else 0.5,
        damping_rounds=2 if scf.damping > 0 else 0,
        level_shift=scf.level_shift if scf.level_shift > 0 else 0.25,
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


def _optimization_warnings(optimization: object) -> tuple[str, ...]:
    """Предупреждения оптимизации — пользователь обязан увидеть несходимость."""
    from quantumlab.engine.optimizer import OptimizationResult

    assert isinstance(optimization, OptimizationResult)
    if optimization.converged:
        return ()
    return (
        f"Оптимизация геометрии не сошлась за {optimization.steps} шагов: "
        f"max|F| = {optimization.max_force:.3e} э/бор. Приведённая геометрия — "
        "последняя достигнутая точка, а не равновесная структура.",
    )


def _report(progress: ProgressReporter | None, percent: float, stage: str, **extra: object) -> None:
    """Сообщает о прогрессе, если приёмник задан."""
    if progress is not None:
        progress.report(percent, stage, **extra)


def _properties(
    rhf: RhfResult,
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
    uhf: UhfResult,
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
    uhf: UhfResult, basis: BasisSet, molecule: Molecule, prepared: PrecomputedIntegrals
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


def _warnings_uhf(uhf: UhfResult, basis: BasisSet, molecule: Molecule) -> tuple[str, ...]:
    """Предупреждения UHF: несошедшийся SCF, схема базиса, начало отсчёта диполя."""
    warnings: list[str] = []
    if not uhf.converged:
        warnings.append(f"SCF не сошёлся за {uhf.iterations} итераций; энергия приведена как есть")
    warning = _angular_scheme_warning(basis)
    if warning:
        warnings.append(warning)
    dipole_warning = _dipole_origin_warning(molecule)
    if dipole_warning:
        warnings.append(dipole_warning)
    return tuple(warnings)


def _quality_checks(
    rhf: RhfResult, basis: BasisSet, molecule: Molecule, prepared: PrecomputedIntegrals
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


def _dipole_origin_warning(molecule: Molecule) -> str | None:
    """Предупреждение о начале отсчёта диполя у заряженной системы.

    Дипольный момент нейтральной системы от начала отсчёта не зависит, а у
    заряженной — зависит линейно. Мы считаем его от начала координат, поэтому
    для иона число воспроизводимо, но физический смысл имеет только вместе с
    указанием этой точки. Молча выдать его как «дипольный момент» нельзя.
    """
    if molecule.charge == 0:
        return None
    return (
        f"Система заряжена (q = {molecule.charge:+d}): дипольный момент зависит от "
        "начала отсчёта и приведён относительно начала координат."
    )


def _angular_scheme_warning(basis: BasisSet) -> str | None:
    """Предупреждение о декартовой схеме, если базис опубликован в сферической.

    Общая для RHF и UHF: текст один, и расхождение в формулировках означало бы,
    что пользователь видит разное предупреждение для одного и того же базиса.
    """
    if basis_angular_scheme(basis.name) == "cartesian":
        return None
    return (
        f"Базис {basis.name} опубликован в сферической схеме, расчёт выполнен в "
        "декартовой. Энергия отличается от табличных значений примерно на 1e-4 Eh."
    )


def _warnings(rhf: RhfResult, basis: BasisSet, molecule: Molecule) -> tuple[str, ...]:
    """Предупреждения, которые обязан увидеть пользователь."""
    warnings: list[str] = []
    if not rhf.converged:
        warnings.append(f"SCF не сошёлся за {rhf.iterations} итераций; энергия приведена как есть")
    warning = _angular_scheme_warning(basis)
    if warning:
        warnings.append(warning)
    dipole_warning = _dipole_origin_warning(molecule)
    if dipole_warning:
        warnings.append(dipole_warning)
    return tuple(warnings)
