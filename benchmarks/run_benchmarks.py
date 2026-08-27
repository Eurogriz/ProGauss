"""Бенчмарки (§27 ТЗ, проект — docs/architecture/09).

Отвечают на вопрос **«считаем ли мы быстро?»** — в отличие от `verification/`,
который отвечает на вопрос «считаем ли мы правильно?».

Правило формулировок (§27 ТЗ): сравнение со сторонним пакетом допускается
только при совпадении метода, базиса, сетки, порогов и версии. Ни одного такого
измерения здесь нет, поэтому скрипт сравнивает нас **с нашими же эталонными
конфигурациями** из `benchmarks/reference/` и публикует абсолютные числа вместе
с описанием окружения.

Методика (§9.2): первый прогон отбрасывается как прогрев, в отчёт идут медиана
и разброс, каждый кейс выполняется в отдельном процессе — иначе пиковая память
предыдущего кейса попала бы в измерение следующего.

Запуск::

    python benchmarks/run_benchmarks.py                  # все наборы
    python benchmarks/run_benchmarks.py --suite small
    python benchmarks/run_benchmarks.py --update-reference small
    python benchmarks/run_benchmarks.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CASES_FILE = Path(__file__).resolve().parent / "cases.yaml"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

#: Относительная деградация wall time, после которой прогон считается красным.
WALL_TOLERANCE = 0.10


# --------------------------------------------------------------------------- #
# Схема
# --------------------------------------------------------------------------- #
class BenchmarkCase(BaseModel):
    """Один бенчмарк: расчёт, стоимость которого измеряется."""

    model_config = ConfigDict(extra="forbid")

    id: str
    suite: str
    molecule: str
    spec: dict[str, Any]
    measure_gradient: bool = False
    runs: int = Field(default=5, ge=2)
    warmup: int = Field(default=1, ge=0)
    blocked_reason: str | None = None


class ReferenceEntry(BaseModel):
    """Эталонное значение одной метрики с допуском."""

    model_config = ConfigDict(extra="forbid")

    value: float
    tolerance: float = Field(ge=0.0)


class Reference(BaseModel):
    """Эталонная конфигурация набора: железо, версии, измеренные метрики."""

    model_config = ConfigDict(extra="forbid")

    suite: str
    measured_at: str
    hardware: str
    versions: dict[str, str]
    cases: dict[str, dict[str, ReferenceEntry]]


@dataclass(slots=True)
class Measurement:
    """Один прогон одного кейса."""

    wall_s: float
    cpu_s: float
    peak_rss_mb: float


@dataclass(slots=True)
class Summary:
    """Сводка по кейсу: медиана и разброс серии прогонов."""

    case_id: str
    runs: int
    wall_s: float
    wall_spread_s: float
    cpu_s: float
    peak_rss_mb: float
    scf_iterations: int
    stage_wall_s: dict[str, float]


# --------------------------------------------------------------------------- #
# Окружение
# --------------------------------------------------------------------------- #
def _blas_name() -> str:
    """Имя BLAS из сборки NumPy — без него сравнение чисел бессмысленно.

    ``np.__config__.show()`` печатает конфиг в stdout и возвращает ``None``,
    поэтому используется ``np.show_config(mode="dicts")``; на старых версиях
    NumPy такого режима нет, и тогда имя честно остаётся неизвестным.
    """
    try:
        info = np.show_config(mode="dicts")
    except TypeError:
        # NumPy до 1.24 не принимает mode — имя BLAS тогда неизвестно.
        return "unknown"
    dependencies = info.get("Build Dependencies")
    entry = dependencies.get("blas") if isinstance(dependencies, dict) else None
    if isinstance(entry, dict) and entry.get("name"):
        version = entry.get("version")
        return f"{entry['name']} {version}" if version else str(entry["name"])
    return "unknown"


def describe_environment() -> dict[str, Any]:
    """Описание железа и версий — без него число невоспроизводимо."""
    git_sha = "unknown"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            git_sha = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    blas = _blas_name()

    return {
        "hardware": (
            f"{platform.machine()}, {os.cpu_count() or '?'} CPU, "
            f"{platform.system()} {platform.release()}"
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas": blas,
            "git": git_sha,
        },
    }


# --------------------------------------------------------------------------- #
# Измерение
# --------------------------------------------------------------------------- #
def _run_calculation(case: BenchmarkCase) -> dict[str, object]:
    """Выполняет расчёт и возвращает времена этапов и итерации SCF.

    Итерации снимаются с прогревающего SCF, а не отдельным расчётом: для
    оптимизации лишний полный прогон стоил бы столько же, сколько измерение.
    """
    from quantumlab.domain.molecule import Molecule
    from quantumlab.domain.spec import (
        CalculationSpec,
        MethodSpec,
        OptimizationSpec,
        Task,
        TheoryFamily,
    )
    from quantumlab.engine.basis import build_basis
    from quantumlab.engine.contracts import EngineRequest
    from quantumlab.engine.gradients import rhf_gradient
    from quantumlab.engine.reference import ReferenceEngine
    from quantumlab.engine.scf import run_rhf

    molecule = Molecule.from_xyz(
        (REPO_ROOT / case.molecule).read_text(encoding="utf-8"), name=case.id
    )
    method_raw = dict(case.spec["method"])
    theory = TheoryFamily(str(method_raw.pop("theory")).lower())
    method = MethodSpec(theory=theory, **method_raw)
    spec = CalculationSpec(
        task=Task(case.spec["task"]),
        method=method,
        optimization=OptimizationSpec(**case.spec.get("optimization", {})),
    )

    basis = build_basis(method.basis, molecule)
    scf = run_rhf(basis, molecule)
    payload: dict[str, object] = {"scf_iterations": scf.iterations}
    if case.measure_gradient:
        rhf_gradient(basis, molecule, scf)
        payload["stages"] = {}
        return payload

    result = ReferenceEngine().run(
        EngineRequest(job_id=f"benchmark-{case.id}", molecule=molecule, spec=spec)
    )
    payload["stages"] = {record.stage: record.wall_seconds for record in result.timings}
    return payload


def _worker(case_id: str) -> int:
    """Тело дочернего процесса: печатает измеренные величины в виде JSON."""
    cases = {item.id: item for item in load_cases()}
    print(json.dumps(_run_calculation(cases[case_id])))
    return 0


def measure(case: BenchmarkCase) -> tuple[Summary, dict[str, float]]:
    """Серия прогонов кейса в отдельных процессах; прогрев отбрасывается."""
    """Серия прогонов в отдельных процессах; прогрев отбрасывается."""
    measurements: list[Measurement] = []
    stages: dict[str, float] = {}
    iterations = -1

    for attempt in range(case.warmup + case.runs):
        started_wall = time.perf_counter()
        # CPU-время берётся из RUSAGE_CHILDREN: расчёт выполняется в дочернем
        # процессе, поэтому time.process_time() родителя его не увидит.
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", case.id],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        wall = time.perf_counter() - started_wall
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
        if completed.returncode != 0:
            tail = completed.stderr[-800:]
            msg = f"кейс {case.id} завершился с кодом {completed.returncode}: {tail}"
            raise RuntimeError(msg)

        # ru_maxrss — монотонный максимум по всем завершённым потомкам, поэтому
        # каждый кейс и запускается отдельным процессом: иначе в измерение
        # попала бы память предыдущего кейса.
        peak_rss_mb = after.ru_maxrss / 1024.0  # Linux: КБ → МБ

        if attempt < case.warmup:
            continue
        if completed.stdout.strip():
            payload = json.loads(completed.stdout.strip())
            stages = {str(k): float(v) for k, v in payload.get("stages", {}).items()}
            iterations = int(payload.get("scf_iterations", -1))
        measurements.append(Measurement(wall_s=wall, cpu_s=cpu, peak_rss_mb=peak_rss_mb))
    if not measurements:
        msg = f"кейс {case.id}: ни одного засчитанного прогона (warmup={case.warmup})"
        raise RuntimeError(msg)

    walls = [item.wall_s for item in measurements]
    summary = Summary(
        case_id=case.id,
        runs=len(measurements),
        wall_s=median(walls),
        wall_spread_s=max(walls) - min(walls),
        cpu_s=median(item.cpu_s for item in measurements),
        peak_rss_mb=max(item.peak_rss_mb for item in measurements),
        scf_iterations=iterations,
        stage_wall_s=stages,
    )
    return summary, stages


# --------------------------------------------------------------------------- #
# Сравнение с эталоном
# --------------------------------------------------------------------------- #
def load_reference(suite: str) -> Reference | None:
    """Эталонная конфигурация набора; ``None``, если её ещё не записывали."""
    path = REFERENCE_DIR / f"{suite}.json"
    if not path.exists():
        return None
    return Reference.model_validate_json(path.read_text(encoding="utf-8"))


def compare(summary: Summary, reference: Reference | None) -> list[str]:
    """Строки расхождений с эталоном; деградация больше допуска помечается."""
    if reference is None:
        return ["    эталона нет — прогон только записывает числа"]
    entries = reference.cases.get(summary.case_id)
    if entries is None:
        return [f"    в эталоне нет кейса {summary.case_id}"]

    lines: list[str] = []
    for metric, obtained in (
        ("wall_s", summary.wall_s),
        ("cpu_s", summary.cpu_s),
        ("peak_rss_mb", summary.peak_rss_mb),
        ("scf_iterations", float(summary.scf_iterations)),
    ):
        entry = entries.get(metric)
        if entry is None or obtained < 0:
            continue
        if entry.value == 0:
            continue
        relative = (obtained - entry.value) / entry.value
        flag = "✓" if relative <= entry.tolerance else "✗ ДЕГРАДАЦИЯ"
        lines.append(
            f"    {metric}: {obtained:.4g} (эталон {entry.value:.4g}, "
            f"{relative:+.1%}, допуск {entry.tolerance:+.0%}) {flag}"
        )
    return lines


def degraded(summary: Summary, reference: Reference | None) -> bool:
    """True, если wall time вышел за допуск эталона."""
    if reference is None:
        return False
    entry = reference.cases.get(summary.case_id, {}).get("wall_s")
    if entry is None or entry.value <= 0:
        return False
    return (summary.wall_s - entry.value) / entry.value > entry.tolerance


# --------------------------------------------------------------------------- #
# Точка входа
# --------------------------------------------------------------------------- #
def load_cases(suites: tuple[str, ...] = ()) -> list[BenchmarkCase]:
    """Читает кейсы; ошибка схемы прерывает запуск."""
    raw = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for item in raw["cases"]:
        try:
            case = BenchmarkCase.model_validate(item)
        except ValidationError as error:
            msg = f"некорректный кейс бенчмарка:\n{error}"
            raise ValueError(msg) from error
        if suites and case.suite not in suites:
            continue
        cases.append(case)
    return cases


def _reference_metrics(item: Summary) -> dict[str, ReferenceEntry]:
    """Метрики кейса для эталона; число итераций — только если оно есть."""
    entries = {
        "wall_s": ReferenceEntry(value=round(item.wall_s, 4), tolerance=WALL_TOLERANCE),
        "cpu_s": ReferenceEntry(value=round(item.cpu_s, 4), tolerance=WALL_TOLERANCE),
        "peak_rss_mb": ReferenceEntry(value=round(item.peak_rss_mb, 1), tolerance=0.25),
    }
    if item.scf_iterations >= 0:
        entries["scf_iterations"] = ReferenceEntry(value=float(item.scf_iterations), tolerance=0.0)
    return entries


def write_reference(suite: str, summaries: list[Summary]) -> Path:
    """Записывает эталон по текущим измерениям (только на зафиксированном железе)."""
    environment = describe_environment()
    reference = Reference(
        suite=suite,
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        hardware=environment["hardware"],
        versions=environment["versions"],
        cases={item.case_id: _reference_metrics(item) for item in summaries},
    )
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCE_DIR / f"{suite}.json"
    path.write_text(reference.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Точка входа: 0 если деградации нет, 1 если wall time вышел за допуск."""
    parser = argparse.ArgumentParser(description="Бенчмарки QuantumLab (§27 ТЗ)")
    parser.add_argument("--suite", action="append", default=[], help="только указанный набор")
    parser.add_argument("--worker", help="служебный режим: выполнить один кейс и выйти")
    parser.add_argument("--update-reference", metavar="SUITE", help="перезаписать эталон")
    parser.add_argument("--list", action="store_true", help="перечислить кейсы и выйти")
    arguments = parser.parse_args(argv)

    if arguments.worker:
        return _worker(arguments.worker)

    suites = tuple(arguments.suite)
    cases = load_cases(suites)

    if arguments.list:
        for case in cases:
            note = f"  [ЗАБЛОКИРОВАН: {case.blocked_reason}]" if case.blocked_reason else ""
            print(f"{case.suite:8s} {case.id}{note}")
        return 0

    if not cases:
        print("нет кейсов для выбранных наборов")
        return 1

    environment = describe_environment()
    print(f"Железо: {environment['hardware']}")
    print(f"Версии: {json.dumps(environment['versions'], ensure_ascii=False)}")
    print()

    failures: list[str] = []
    collected: dict[str, list[Summary]] = {}

    for case in cases:
        print(f"[{case.suite}] {case.id}")
        if case.blocked_reason:
            print(f"    ЗАБЛОКИРОВАН: {case.blocked_reason}")
            print()
            continue

        summary, stages = measure(case)
        collected.setdefault(case.suite, []).append(summary)
        print(
            f"    wall: {summary.wall_s:.3f} с (медиана {summary.runs} прогонов, "
            f"разброс {summary.wall_spread_s:.3f} с)"
        )
        print(f"    cpu: {summary.cpu_s:.3f} с   пиковая память: {summary.peak_rss_mb:.0f} МБ")
        if summary.scf_iterations >= 0:
            print(f"    итераций SCF: {summary.scf_iterations}")
        for stage, seconds in stages.items():
            print(f"      этап {stage}: {seconds:.3f} с")
        for line in compare(summary, load_reference(case.suite)):
            print(line)
        if degraded(summary, load_reference(case.suite)):
            failures.append(case.id)
        print()

    if arguments.update_reference:
        summaries = collected.get(arguments.update_reference, [])
        if not summaries:
            print(f"нет измерений для набора {arguments.update_reference}")
            return 1
        path = write_reference(arguments.update_reference, summaries)
        print(f"Эталон записан: {path.relative_to(REPO_ROOT)}")

    if failures:
        print(f"ДЕГРАДАЦИЯ выше допуска: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
