"""Проверки обвязки бенчмарков (§27 ТЗ).

Сами прогоны долгие и зависят от железа, поэтому здесь проверяется не
скорость, а то, что измерение вообще корректно устроено: схема кейсов,
соответствие эталона, честность заблокированных кейсов и работа детектора
деградации.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("yaml", reason="бенчмарки требуют PyYAML (dev-зависимость)")


def _load() -> ModuleType:
    path = REPO_ROOT / "benchmarks" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("benchmark_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = _load()


def test_cases_are_valid_and_have_suites() -> None:
    cases = runner.load_cases()
    assert cases, "cases.yaml пуст"
    assert all(case.suite for case in cases)
    assert {case.suite for case in cases} == {"small", "medium"}


def test_blocked_cases_state_a_reason() -> None:
    """Заблокированный кейс обязан объяснять причину, а не просто отсутствовать.

    Прежнее требование «хотя бы один заблокированный кейс существует» снято:
    оно заставляло бы держать кейс заблокированным и после устранения причины.
    Бензол/6-31G был заблокирован из-за 245 с на сборку ERI; пакетная сборка
    квартетов сократила её до ~12 с, и кейс измеряется наравне с остальными.
    """
    cases = runner.load_cases()
    for case in cases:
        if case.blocked_reason:
            assert len(case.blocked_reason) > 30, case.id
            assert case.suite == "medium"
    measurable = {case.id for case in cases if not case.blocked_reason}
    assert "benzene-rhf-631g-sp" in measurable


def test_reference_matches_cases_and_is_parseable() -> None:
    """Эталон описывает ровно измеримые кейсы своего набора."""
    reference = runner.load_reference("small")
    assert reference is not None, "benchmarks/reference/small.json отсутствует"
    assert reference.suite == "small"
    assert reference.hardware and reference.versions.get("numpy")

    measurable = {case.id for case in runner.load_cases(("small",)) if not case.blocked_reason}
    assert set(reference.cases) == measurable


def test_reference_values_are_plausible() -> None:
    """Эталон не содержит нулей и отрицательных величин — это признак сбоя записи."""
    reference = runner.load_reference("small")
    assert reference is not None
    for case_id, metrics in reference.cases.items():
        assert "wall_s" in metrics, case_id
        for name, entry in metrics.items():
            assert entry.value > 0, f"{case_id}.{name}"
            assert entry.tolerance >= 0, f"{case_id}.{name}"


def test_degradation_detector_catches_slowdown() -> None:
    """Замедление больше допуска — ненулевой выход, а не «шум»."""
    reference = runner.load_reference("small")
    baseline = reference.cases["water-rhf-sto3g-sp"]["wall_s"].value

    def summary(wall: float) -> object:
        return runner.Summary("water-rhf-sto3g-sp", 3, wall, 0.01, wall, 49.0, 8, {})

    assert not runner.degraded(summary(baseline), reference)
    assert runner.degraded(summary(baseline * 2.0), reference)
    lines = "\n".join(runner.compare(summary(baseline * 2.0), reference))
    assert "ДЕГРАДАЦИЯ" in lines


def test_tolerance_floor_absorbs_stand_drift_but_not_a_real_slowdown() -> None:
    """Порог допуска глушит дрейф стенда, а не настоящие регрессии.

    Относительный допуск 10 % на кейсе в 0.3 c — это 30 мс, меньше шума
    стенда: повторный прогон того же кода давал систематическое смещение
    до 49 мс. Порог ``TOLERANCE_FLOOR_S`` снимает ложные срабатывания, но
    обязан оставлять видимым замедление, которое больше него.
    """
    reference = runner.load_reference("small")
    assert reference is not None
    baseline = reference.cases["water-rhf-sto3g-sp"]["wall_s"].value

    def summary(wall: float) -> object:
        return runner.Summary("water-rhf-sto3g-sp", 3, wall, 0.01, wall, 49.0, 8, {})

    drift = baseline + 0.9 * runner.TOLERANCE_FLOOR_S
    assert not runner.degraded(summary(drift), reference)
    assert "✓" in "\n".join(runner.compare(summary(drift), reference))

    slow = baseline + 1.5 * runner.TOLERANCE_FLOOR_S
    assert runner.degraded(summary(slow), reference)
    assert "ДЕГРАДАЦИЯ" in "\n".join(runner.compare(summary(slow), reference))


def test_report_and_exit_code_never_disagree() -> None:
    """Строка отчёта и код выхода судят о деградации по одному и тому же числу.

    Проверка по сетке отклонений: если «ДЕГРАДАЦИЯ» напечатана, процесс обязан
    завершиться ненулевым кодом, и наоборот. Разъехавшиеся мнения дали бы
    отчёт с галочкой и красный CI (или наоборот).
    """
    reference = runner.load_reference("small")
    assert reference is not None
    for case_id, metrics in reference.cases.items():
        baseline = metrics["wall_s"].value
        # Остальные метрики берутся из эталона, чтобы сетка разворачивалась
        # ровно по оси времени: иначе расхождение итераций подмешалось бы в
        # проверку и она прошла бы «случайно».
        cpu = metrics["cpu_s"].value
        rss = metrics["peak_rss_mb"].value
        iterations = int(metrics["scf_iterations"].value)
        for factor in (0.5, 0.9, 1.0, 1.05, 1.2, 1.5, 2.0, 4.0):
            wall = baseline * factor
            summary = runner.Summary(case_id, 3, wall, 0.01, cpu, rss, iterations, {})
            printed = "ДЕГРАДАЦИЯ" in "\n".join(runner.compare(summary, reference))
            assert printed == runner.degraded(summary, reference), (case_id, factor)


def test_changed_iteration_count_is_a_degradation_for_the_exit_code() -> None:
    """Расхождение числа итераций красит и отчёт, и код выхода.

    Раньше ``degraded`` смотрел только на ``wall_s``, поэтому изменившееся
    число итераций печатало «ДЕГРАДАЦИЯ» и при этом процесс завершался нулём.
    Итерации — содержательный сигнал: молча ослабленный порог сходимости
    даёт ту же энергию до 8 знаков, но меньше итераций.
    """
    reference = runner.load_reference("small")
    assert reference is not None
    case_id = "water-rhf-631g-sp"
    metrics = reference.cases[case_id]
    wall = metrics["wall_s"].value
    baseline = runner.Summary(
        case_id,
        3,
        wall,
        0.01,
        metrics["cpu_s"].value,
        metrics["peak_rss_mb"].value,
        int(metrics["scf_iterations"].value),
        {},
    )
    assert not runner.degraded(baseline, reference)

    changed = runner.Summary(
        case_id,
        3,
        wall,
        0.01,
        metrics["cpu_s"].value,
        metrics["peak_rss_mb"].value,
        int(metrics["scf_iterations"].value) + 1,
        {},
    )
    assert runner.degraded(changed, reference)
    assert "ДЕГРАДАЦИЯ" in "\n".join(runner.compare(changed, reference))


def test_time_floor_does_not_leak_into_memory_and_iterations() -> None:
    """Порог действует только на время: память и итерации судятся относительно.

    Иначе кейс с 50 МБ получил бы допуск 0.1 МБ — формально «ниже шума»,
    фактически бессмысленный, — а расхождение числа итераций перестало бы
    ловиться на больших кейсах.
    """
    reference = runner.load_reference("small")
    assert reference is not None
    for metrics in reference.cases.values():
        for metric, entry in metrics.items():
            allowed = runner._allowance(metric, entry)
            if metric in runner._TIME_METRICS:
                assert allowed >= runner.TOLERANCE_FLOOR_S
            else:
                assert allowed == entry.tolerance * entry.value


def test_environment_description_is_complete() -> None:
    """Без описания железа и версий число невоспроизводимо (§27 ТЗ)."""
    environment = runner.describe_environment()
    assert environment["hardware"]
    versions = environment["versions"]
    for key in ("python", "numpy", "blas", "git"):
        assert versions[key], key
    assert versions["blas"] != "unknown", "имя BLAS не определено"


def test_reference_json_is_valid_json_on_disk() -> None:
    path = REPO_ROOT / "benchmarks" / "reference" / "small.json"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["suite"] == "small"
    assert parsed["cases"]
