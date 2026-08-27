"""Быстрая часть верификационного набора в общем прогоне (§26 ТЗ).

Полный набор (``verification/run_verification.py``) идёт ~2.5 минуты, потому
что уровни L2 и L5 требуют десятков SCF-расчётов; в общий прогон он не входит.
Здесь выполняются уровни, которые укладываются в секунды: L1 (энергии),
L7 (симметрия) и L9 (воспроизводимость).

Кейсы берутся из тех же YAML-файлов, что и автономный запуск, — двух копий
эталонов не существует.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_CASES = REPO_ROOT / "verification" / "cases"

pytest.importorskip("yaml", reason="верификационный набор требует PyYAML (dev-зависимость)")


def _load_runner() -> ModuleType:
    """Загружает раннер по пути: ``verification`` не является пакетом."""
    path = REPO_ROOT / "verification" / "run_verification.py"
    spec = importlib.util.spec_from_file_location("verification_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verification_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

#: Уровни, для которых есть кейсы. Фиксированный перечень нужен, чтобы набор
#: не «сжимался» молча: новый реализованный расчёт обязан получить кейс.
KNOWN_LEVELS = ("L1", "L2", "L5", "L7", "L9")


def _slow_ids() -> list[str]:
    return [case.id for case in runner.load_cases() if case.slow]


def test_fast_cases_run_and_pass() -> None:
    """Все кейсы без флага ``slow`` проходят: это прогон на каждый коммит."""
    outcomes = runner.run(include_slow=False)
    failed = [item.case_id for item in outcomes if not item.passed]
    assert len(outcomes) >= 5, "быстрый набор подозрительно мал"
    assert not failed, f"провалены кейсы: {failed}"


def test_known_levels_are_covered() -> None:
    present = {case.level for case in runner.load_cases(include_slow=False)}
    assert present == set(KNOWN_LEVELS) - {"L5"}, "в быстром наборе пропал уровень"


def test_unknown_field_in_case_is_rejected() -> None:
    """Опечатка в имени поля кейса — ошибка, а не молча пропущенная проверка."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        runner.Case.model_validate(
            {
                "id": "broken",
                "level": "L1",
                "molecule": "tests/fixtures/water.xyz",
                "spec": {"task": "single_point", "method": {"theory": "hf", "basis": "sto-3g"}},
                "expectation": {"energy_hartree": {"value": 0.0, "tol": 1e-8, "source": "x"}},
            }
        )


def test_cases_directory_is_not_empty() -> None:
    assert any(_CASES.rglob("*.yaml")), "каталог verification/cases пуст"


@pytest.mark.slow
@pytest.mark.parametrize("case_id", _slow_ids())
def test_slow_cases_run_and_pass(case_id: str) -> None:
    """Дорогие кейсы (RHF/cc-pVDZ, оптимизация, FD-градиент воды).

    Обязательны перед релизом, но в быстрый прогон не входят: вода/cc-pVDZ
    на одной точке занимает ~110 с из-за скалярной сборки ERI.
    """
    outcomes = runner.run(only=(case_id,))
    assert len(outcomes) == 1, case_id
    assert outcomes[0].passed, outcomes[0].error or outcomes[0].lines
