"""Контракт REST API: валидность OpenAPI и согласованность с доменными enum.

Тест защищает от главного класса дефектов в API-первом продукте: схема в
``api/openapi/v1.yaml`` живёт отдельно от кода и молча расходится с ним.
Здесь сверяются перечисления статусов, задач и профилей.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from openapi_spec_validator import validate

from quantumlab.domain.spec import PrecisionProfile, Task
from quantumlab.domain.spec import Task as DomainTask
from quantumlab.jobs.state_machine import JobStatus

SPEC_PATH = Path(__file__).resolve().parents[1] / "api" / "openapi" / "v1.yaml"

pytest.importorskip("openapi_spec_validator", reason="контракт проверяется в dev-окружении")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    with SPEC_PATH.open(encoding="utf-8") as handle:
        document: object = yaml.safe_load(handle)
    assert isinstance(document, dict)
    return cast("dict[str, Any]", document)


def test_openapi_document_is_valid(spec: dict[str, Any]) -> None:
    validate(spec)
    assert spec["openapi"].startswith("3.1")
    assert spec["info"]["version"] == "v1"


def test_server_prefix_is_versioned(spec: dict[str, Any]) -> None:
    assert spec["servers"][0]["url"] == "/api/v1"


def test_job_status_enum_matches_domain(spec: dict[str, Any]) -> None:
    declared = set(spec["components"]["schemas"]["JobStatus"]["enum"])
    assert declared == {status.value for status in JobStatus}


def test_task_enum_matches_domain(spec: dict[str, Any]) -> None:
    declared = set(spec["components"]["schemas"]["Task"]["enum"])
    assert declared == {task.value for task in Task}


def test_profile_enum_matches_domain(spec: dict[str, Any]) -> None:
    declared = set(spec["components"]["schemas"]["PrecisionProfile"]["enum"])
    assert declared == {profile.value for profile in PrecisionProfile}


def test_every_task_is_addressable_through_plan(spec: dict[str, Any]) -> None:
    plan_schema = spec["components"]["schemas"]["PlanRequest"]["properties"]["task"]
    assert plan_schema["$ref"].endswith("/Task")
    assert DomainTask.OPTIMIZATION.value in spec["components"]["schemas"]["Task"]["enum"]


def test_error_response_carries_diagnosis_fields(spec: dict[str, Any]) -> None:
    problem = spec["components"]["schemas"]["Problem"]
    assert {"type", "title", "status", "code"} <= set(problem["required"])
    for field in ("detail", "hint", "tried", "actions"):
        assert field in problem["properties"], f"в Problem нет поля {field}"


def test_capabilities_endpoint_exposes_availability(spec: dict[str, Any]) -> None:
    capability = spec["components"]["schemas"]["Capability"]
    assert capability["properties"]["availability"]["enum"] == [
        "implemented",
        "partial",
        "not_implemented",
    ]
    assert "/capabilities" in spec["paths"]


def test_health_endpoints_are_unauthenticated(spec: dict[str, Any]) -> None:
    for path in ("/health", "/ready"):
        assert spec["paths"][path]["get"].get("security") == [], f"{path} должен быть открытым"


def test_language_header_is_documented(spec: dict[str, Any]) -> None:
    header = spec["components"]["parameters"]["AcceptLanguage"]
    assert header["schema"]["default"] == "ru"
    assert header["schema"]["enum"] == ["ru", "en"]
