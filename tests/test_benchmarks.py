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
