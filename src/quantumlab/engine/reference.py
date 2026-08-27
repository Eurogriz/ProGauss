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
from quantumlab.engine.registry import CapabilityRegistry, default_registry
from quantumlab.engine.scf import ScfResult as RhfResult
from quantumlab.engine.scf import ScfSettings, canonical_orthogonalizer, run_rhf
from quantumlab.version import __version__

#: Имя ядра — попадает в отпечаток расчёта и в журнал.
ENGINE_NAME = "quantumlab-reference"

#: Backend, который фактически используется: один поток, dense float64.
ENGINE_BACKEND = "numpy-dense-cpu"

#: Точность, с которой tr(D·S) должен равняться числу электронов.
_ELECTRON_COUNT_TOLERANCE = 1e-6

#: Отклонение −V/T от 2, при котором теорема вириала считается выполненной.
#: На неоптимизированной геометрии отклонение ожидаемо, поэтому превышение
#: порога даёт ``warn``, а не ``fail``.
_VIRIAL_TOLERANCE = 0.02

#: Точность выполнения D′² = 2D′ в ортогональном базисе.
_IDEMPOTENCY_TOLERANCE = 1e-6


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
        return (Task.SINGLE_POINT.value,)

    # ------------------------------------------------------------------ #
    # Основной вход
    # ------------------------------------------------------------------ #
    def run(
        self, request: EngineRequest, *, progress: ProgressReporter | None = None
    ) -> CalculationResult:
        """Выполняет расчёт в точке и возвращает результат с проверками качества.

        Любая неподдерживаемая комбинация (задача, метод, спин, базис)
        отклоняется штатной ошибкой до начала вычислений — пользователь
        получает «этот метод пока недоступен», а не правдоподобное число.
        """
        spec = request.spec
        basis_name = self.assert_supported(spec)

        timings: list[TimingRecord] = []

        started = time.perf_counter()
        basis = build_basis(basis_name, request.molecule)
        timings.append(_timing("basis", started))
        _report(progress, 5.0, "basis", functions=basis.n_functions)

        started = time.perf_counter()
        overlap = integrals.build_overlap(basis, request.molecule)
        core = integrals.build_core_hamiltonian(basis, request.molecule)
        integrals.build_electron_repulsion(basis, request.molecule)
        dipole_integrals = integrals.build_dipole_integrals(basis, request.molecule)
        timings.append(_timing("integrals", started))
        _report(progress, 45.0, "integrals")

        started = time.perf_counter()
        rhf = run_rhf(basis, request.molecule, _scf_settings(spec))
        timings.append(_timing("scf", started))
        _report(progress, 85.0, "scf", iterations=rhf.iterations, converged=rhf.converged)

        started = time.perf_counter()
        properties = _properties(rhf, request.molecule, dipole_integrals)
        checks = _quality_checks(rhf, basis, request.molecule, overlap)
        timings.append(_timing("properties", started))
        _report(progress, 100.0, "properties")

        fingerprint = build_fingerprint(
            spec=spec,
            molecule=request.molecule,
            software_version=__version__,
            engine_version=self.version,
            hardware={"backend": ENGINE_BACKEND, "threads": str(request.threads)},
        )
        del core  # построен для полноты протокола; в RHF используется внутри SCF

        return CalculationResult(
            job_id=request.job_id,
            spec=spec,
            fingerprint=fingerprint,
            energy_hartree=rhf.total_energy,
            scf_iterations=rhf.iterations,
            converged=rhf.converged,
            homo_energy_hartree=properties.homo,
            lumo_energy_hartree=properties.lumo,
            gap_hartree=properties.gap,
            dipole_debye=properties.dipole_debye,
            orbitals=properties.orbitals,
            quality_checks=checks,
            timings=tuple(timings),
            warnings=_warnings(rhf, basis),
            environment=_environment(),
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
        if spec.task is not Task.SINGLE_POINT:
            self._registry.assert_available(f"task:{spec.task.value}")
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


def _quality_checks(
    rhf: RhfResult, basis: BasisSet, molecule: Molecule, overlap: np.ndarray
) -> tuple[QualityCheck, ...]:
    """Проверки, которые движок выполняет над собственным результатом (§26 ТЗ).

    Проверки нужны не для отчёта, а чтобы расхождение между математическими
    тождествами и фактическим результатом не прошло незамеченным.
    """
    electron_count = float(np.trace(rhf.density @ overlap))
    expected = molecule.n_electrons

    kinetic = float(np.sum(rhf.density * integrals.build_kinetic(basis, molecule)))
    virial_ratio = (kinetic - rhf.total_energy) / kinetic if kinetic != 0.0 else float("nan")

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
            name_key="virial_ratio",
            verdict=(
                QualityVerdict.PASS
                if abs(virial_ratio - 2.0) < _VIRIAL_TOLERANCE
                else QualityVerdict.WARNING
            ),
            detail=(
                f"−V/T = {virial_ratio:.6f} (в равновесной геометрии равно 2); "
                "отклонение ожидаемо, если геометрия не оптимизирована"
            ),
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


def _warnings(rhf: RhfResult, basis: BasisSet) -> tuple[str, ...]:
    """Предупреждения, которые обязан увидеть пользователь."""
    warnings: list[str] = []
    if not rhf.converged:
        warnings.append(f"SCF не сошёлся за {rhf.iterations} итераций; энергия приведена как есть")
    if basis_angular_scheme(basis.name) != "cartesian":
        warnings.append(
            f"Базис {basis.name} опубликован в сферической схеме, расчёт выполнен в "
            "декартовой. Энергия отличается от табличных значений примерно на 1e-4 Eh."
        )
    return tuple(warnings)
