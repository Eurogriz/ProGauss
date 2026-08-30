"""Проверки REST API по контракту ``api/openapi/v1.yaml``.

Поднимают приложение на временном каталоге: ни один тест не пишет в реальный
``data/`` и не зависит от порядка запуска.

Отдельно закреплены два обязательства, которые легко потерять при рефакторинге:

* нереализованные операции отдают ``501`` с кодом ``api.not_implemented``,
  а не прячутся за ``404``;
* план состоит только из того, что система умеет (§54 ТЗ).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quantumlab.server import create_app
from quantumlab.server.app import Services
from quantumlab.server.worker import run_pending_jobs

FIXTURES = Path("tests/fixtures")

#: Срез реестра: категория → список возможностей.
Snapshot = dict[str, list[dict[str, object]]]


@pytest.fixture()
def app(tmp_path: Path) -> FastAPI:
    """Приложение на изолированном каталоге данных."""
    return create_app(tmp_path / "data")


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    """HTTP-клиент к приложению из фикстуры ``app``."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def services(app: FastAPI) -> Services:
    """Сервисы приложения — нужны, чтобы вызвать воркер напрямую."""
    return cast("Services", app.state.services)


@pytest.fixture()
def hydrogen_id(client: TestClient) -> str:
    """Молекула водорода в новом проекте; возвращает её идентификатор."""
    project = client.post("/projects", json={"name": "тест"})
    content = (FIXTURES / "hydrogen.xyz").read_text(encoding="utf-8")
    molecule = client.post(
        f"/projects/{project.json()['id']}/molecules",
        json={"name": "водород", "content": content},
    )
    return str(molecule.json()["id"])


# --------------------------------------------------------------------------- #
# Служебные эндпоинты
# --------------------------------------------------------------------------- #
def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["checks"]["storage"] is True


def _availability(snapshot: Snapshot, category: str, identifier: str) -> str:
    """Статус возможности в срезе; KeyError, если её там нет."""
    for capability in snapshot[category]:
        if capability["id"] == identifier:
            return str(capability["availability"])
    raise KeyError(identifier)


def test_capabilities_reports_registry(client: TestClient) -> None:
    body = client.get("/capabilities").json()
    assert _availability(body, "method", "method:hf") == "partial"
    # DFT реализован частично (только LDA-функционал SVWN, только энергия в
    # точке), поэтому partial, а не not_implemented и не implemented.
    assert _availability(body, "method", "method:dft") == "partial"
    assert _availability(body, "functional", "functional:svwn") == "partial"
    assert _availability(body, "functional", "functional:b3lyp") == "partial"
    assert _availability(body, "functional", "functional:tpssh") == "not_implemented"
    # Системы координат и спин — не методы: «База методов» в GUI строится отсюда.
    assert "coordinates" in body and "spin" in body
    assert not any(item["id"].startswith(("spin:", "coordinates:")) for item in body["method"])


def test_i18n_catalog_served_per_locale(client: TestClient) -> None:
    ru = client.get("/i18n/ru")
    en = client.get("/i18n/en")
    assert ru.status_code == en.status_code == 200
    assert ru.json()["wizard.step.molecule"] == "1. Молекула"
    assert en.json()["wizard.step.molecule"] == "1. Molecule"
    assert set(ru.json()) == set(en.json())


def test_unknown_locale_is_404(client: TestClient) -> None:
    assert client.get("/i18n/de").status_code == 404


# --------------------------------------------------------------------------- #
# Проекты и структуры
# --------------------------------------------------------------------------- #
def test_project_roundtrip(client: TestClient) -> None:
    created = client.post("/projects", json={"name": "проект"})
    assert created.status_code == 201
    assert created.json()["name"] == "проект"
    listed = client.get("/projects").json()
    assert listed["total"] == 1
    assert listed["items"][0]["id"] == created.json()["id"]


def test_molecule_is_parsed_not_stored_verbatim(client: TestClient) -> None:
    """Имя выводится из формулы, координаты — разобранные числа."""
    project = client.post("/projects", json={"name": "п"}).json()
    body = client.post(
        f"/projects/{project['id']}/molecules",
        json={"content": (FIXTURES / "water.xyz").read_text(encoding="utf-8")},
    ).json()
    assert body["name"] == "H2O"
    assert body["atoms"][0]["symbol"] == "O"
    assert body["atoms"][1]["position"] == [0.7571689334, 0.5865799573, 0.0]


def test_unsupported_format_is_rejected_honestly(client: TestClient) -> None:
    project = client.post("/projects", json={"name": "п"}).json()
    response = client.post(
        f"/projects/{project['id']}/molecules",
        json={"format": "smiles", "content": "O"},
    )
    assert response.status_code == 501
    assert response.json()["code"] == "format.not_implemented"


def test_unknown_molecule_is_404_with_problem(client: TestClient) -> None:
    response = client.get("/molecules/нет-такой")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "catalog.not_found"
    assert body["title"] and body["detail"]


def test_validate_reports_odd_electron_count(client: TestClient) -> None:
    """Нечётное число электронов видно до расчёта: RHF так не считается."""
    project = client.post("/projects", json={"name": "п"}).json()
    molecule = client.post(
        f"/projects/{project['id']}/molecules",
        json={
            "content": (FIXTURES / "hydrogen.xyz").read_text(encoding="utf-8"),
            "charge": 1,
            "multiplicity": 2,
        },
    ).json()
    report = client.post(f"/molecules/{molecule['id']}/validate").json()
    assert report["valid"] is False
    assert report["issues"][0]["code"] == "molecule.invalid_multiplicity"


# --------------------------------------------------------------------------- #
# Подбор параметров
# --------------------------------------------------------------------------- #
def test_plan_never_recommends_unimplemented_method(client: TestClient, hydrogen_id: str) -> None:
    """§54 ТЗ: план обязан состоять из того, что система умеет выполнить."""
    capabilities = client.get("/capabilities").json()
    for profile in ("screening", "standard", "high_accuracy", "research"):
        spec = client.post(
            "/calculations/plan",
            json={"task": "single_point", "profile": profile, "moleculeId": hydrogen_id},
        ).json()["spec"]
        method = spec["method"]
        assert _availability(capabilities, "method", f"method:{method['theory']}") != (
            "not_implemented"
        )
        if method["theory"] == "dft":
            assert (
                _availability(capabilities, "functional", f"functional:{method['functional']}")
                != "not_implemented"
            )


def test_plan_keeps_available_dispersion(client: TestClient, hydrogen_id: str) -> None:
    """Реализованная часть обещания профиля остаётся в плане и видна явно.

    Профиль «Стандартный расчёт» обещает PBE0-D3(BJ): и функционал, и
    дисперсионная поправка реализованы, поэтому план обязан показать обе
    части обещания — «d3bj» в спецификации и отдельным решением в
    обоснованиях (§8 ТЗ). Ветка «недоступная поправка снимается явно»
    проверяется в test_recommend_profiles.py через реестр без D3.
    """
    body = client.post(
        "/calculations/plan",
        json={"task": "single_point", "profile": "standard", "moleculeId": hydrogen_id},
        headers={"Accept-Language": "ru"},
    ).json()
    assert body["spec"]["method"]["theory"] == "dft"
    assert body["spec"]["method"]["functional"] == "pbe0"
    assert body["spec"]["method"]["dispersion"] == "d3bj"
    reasons = [d["text"] for d in body["decisions"] if d["parameter"] == "dispersion"]
    assert reasons == ["Дисперсионная поправка: d3bj"]


def test_plan_is_localized(client: TestClient, hydrogen_id: str) -> None:
    payload = {"task": "single_point", "profile": "standard", "moleculeId": hydrogen_id}
    ru = client.post("/calculations/plan", json=payload, headers={"Accept-Language": "ru"}).json()
    en = client.post("/calculations/plan", json=payload, headers={"Accept-Language": "en"}).json()
    assert "Потому что" in ru["rationale"]
    assert "Because" in en["rationale"]


def test_plan_accepts_camel_and_snake_case(client: TestClient, hydrogen_id: str) -> None:
    """Контракт в camelCase, но собранные вручную тела тоже должны работать."""
    camel = {"task": "single_point", "profile": "standard", "moleculeId": hydrogen_id}
    snake = {"task": "single_point", "profile": "standard", "molecule_id": hydrogen_id}
    assert client.post("/calculations/plan", json=camel).status_code == 200
    assert client.post("/calculations/plan", json=snake).status_code == 200


# --------------------------------------------------------------------------- #
# Задания и воркер
# --------------------------------------------------------------------------- #
def test_job_goes_through_queue_to_result(
    client: TestClient, services: Services, hydrogen_id: str
) -> None:
    """Полный путь: очередь → воркер → результат."""
    spec = client.post(
        "/calculations/plan",
        json={"task": "single_point", "profile": "screening", "moleculeId": hydrogen_id},
    ).json()["spec"]

    accepted = client.post("/jobs", json={"moleculeId": hydrogen_id, "spec": spec})
    assert accepted.status_code == 202
    job_id = accepted.json()["id"]
    assert accepted.json()["status"] == "queued"

    outcomes = run_pending_jobs(services.jobs, services.catalog)
    assert [o.job_id for o in outcomes] == [job_id]

    result = client.get(f"/jobs/{job_id}/result").json()
    # Ответ в camelCase, как объявлено в контракте.
    assert result["energyHartree"] < -1.0
    assert result["scfIterations"] > 0
    assert result["jobId"] == job_id


def test_job_list_filters_by_project(client: TestClient, hydrogen_id: str) -> None:
    spec = client.post(
        "/calculations/plan",
        json={"task": "single_point", "profile": "screening", "moleculeId": hydrogen_id},
    ).json()["spec"]
    job = client.post("/jobs", json={"moleculeId": hydrogen_id, "spec": spec}).json()
    assert client.get("/jobs").json()["total"] == 1
    assert client.get("/jobs", params={"projectId": job["projectId"]}).json()["total"] == 1
    assert client.get("/jobs", params={"projectId": "другой"}).json()["total"] == 0


def test_unimplemented_task_is_rejected_before_queueing(
    client: TestClient, hydrogen_id: str
) -> None:
    response = client.post(
        "/jobs",
        json={
            "moleculeId": hydrogen_id,
            "spec": {"task": "irc", "method": {"theory": "hf", "basis": "sto-3g"}},
        },
    )
    assert response.status_code == 501
    assert response.json()["code"] == "engine.method_not_available"


def test_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/jobs/нет-такой")
    assert response.status_code == 404
    assert response.json()["code"] == "catalog.not_found"


def test_result_before_execution_is_409(client: TestClient, hydrogen_id: str) -> None:
    spec = client.post(
        "/calculations/plan",
        json={"task": "single_point", "profile": "screening", "moleculeId": hydrogen_id},
    ).json()["spec"]
    job_id = client.post("/jobs", json={"moleculeId": hydrogen_id, "spec": spec}).json()["id"]
    assert client.get(f"/jobs/{job_id}/result").status_code == 409


def test_malformed_body_is_422_problem(client: TestClient) -> None:
    response = client.post("/jobs", json={"moleculeId": "x"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation.failed"
    assert body["status"] == 422
    assert body["type"].startswith("https://quantumlab.dev/errors/")


# --------------------------------------------------------------------------- #
# Честность про нереализованное
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/auth/token"),
        ("get", "/auth/me"),
        ("get", "/workers"),
        ("post", "/jobs/some-id/cancel"),
        ("post", "/jobs/some-id/resume"),
        ("get", "/jobs/some-id/logs"),
        ("get", "/jobs/some-id/report"),
        ("get", "/artifacts/some-id"),
    ],
)
def test_unimplemented_routes_answer_501_not_404(
    client: TestClient, method: str, path: str
) -> None:
    """404 означал бы «такого пути нет в API», а контракт его объявляет."""
    response = client.request(method, path)
    assert response.status_code == 501
    body = response.json()
    assert body["code"] == "api.not_implemented"
    assert body["hint"]


def test_unimplemented_message_is_localized(client: TestClient) -> None:
    ru = client.post("/auth/token", headers={"Accept-Language": "ru"}).json()
    en = client.post("/auth/token", headers={"Accept-Language": "en"}).json()
    assert "ещё не готова" in ru["detail"]
    assert "not ready" in en["detail"]
