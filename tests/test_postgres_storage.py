"""PostgreSQL-хранилище против настоящей базы (встроенный сервер ``pgserver``).

Две группы проверок:

* **паритет** — файловое и PostgreSQL-хранилища проходят одни и те же тесты, так
  что потребитель не замечает подмены;
* **то, что файлы дать не могут** — потерянных обновлений нет, очередь выдаёт
  каждое задание ровно одному воркеру, миграции идемпотентны и безопасны при
  параллельном старте, повреждённая строка не читается молча.

База поднимается один раз на сессию, каждый тест получает свежую базу данных.
Задайте ``QUANTUMLAB_TEST_DATABASE_URL``, чтобы использовать внешний сервер.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

psycopg = pytest.importorskip("psycopg")

from fastapi.testclient import TestClient  # noqa: E402

from quantumlab.cli import main  # noqa: E402
from quantumlab.domain.job import Job  # noqa: E402
from quantumlab.domain.molecule import Molecule  # noqa: E402
from quantumlab.domain.spec import CalculationSpec, MethodSpec, Task, TheoryFamily  # noqa: E402
from quantumlab.errors import (  # noqa: E402
    CatalogEntryNotFoundError,
    InvalidJobTransitionError,
    JobCheckpointInvalidError,
    UnsupportedStructureFormatError,
)
from quantumlab.jobs.state_machine import JobStatus  # noqa: E402
from quantumlab.server import create_app  # noqa: E402
from quantumlab.server.app import Services  # noqa: E402
from quantumlab.server.worker import run_pending_jobs  # noqa: E402
from quantumlab.storage.base import JobStore  # noqa: E402
from quantumlab.storage.local_catalog import LocalCatalog  # noqa: E402
from quantumlab.storage.local_jobs import LocalJobStore  # noqa: E402
from quantumlab.storage.postgres import (  # noqa: E402
    MIGRATIONS,
    PostgresCatalog,
    PostgresDatabase,
    PostgresJobStore,
    _redact,
)

FIXTURES = Path(__file__).parent / "fixtures"
HYDROGEN = (FIXTURES / "hydrogen.xyz").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Инфраструктура
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def pg_admin_dsn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """DSN сервера PostgreSQL: внешний из окружения или встроенный pgserver."""
    external = os.environ.get("QUANTUMLAB_TEST_DATABASE_URL")
    if external:
        yield external
        return
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop")
    try:
        yield str(server.get_uri())
    finally:
        server.cleanup()


def _dsn_for(admin_dsn: str, database: str) -> str:
    """DSN той же базы с другим именем (формат URI pgserver и обычный URI)."""
    head, sep, tail = admin_dsn.partition("?")
    base, _, _ = head.rpartition("/")
    return f"{base}/{database}{sep}{tail}"


@pytest.fixture()
def fresh_database(pg_admin_dsn: str) -> str:
    """DSN новой пустой базы данных — по одной на тест."""
    name = f"ql_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg_admin_dsn, autocommit=True) as connection:
        connection.execute(f'CREATE DATABASE "{name}"')
    return _dsn_for(pg_admin_dsn, name)


@pytest.fixture()
def database(fresh_database: str) -> PostgresDatabase:
    return PostgresDatabase(fresh_database)


@pytest.fixture(params=["local", "postgres"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> JobStore:
    """Одно и то же хранилище заданий в двух реализациях."""
    if request.param == "local":
        return LocalJobStore(tmp_path / "jobs")
    return PostgresJobStore(request.getfixturevalue("database"))


def _spec() -> CalculationSpec:
    return CalculationSpec(
        task=Task.SINGLE_POINT, method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g")
    )


def _job(name: str = "job", *, priority: int = 100) -> Job:
    return Job(
        name=name,
        project_id="project",
        owner="tester",
        spec=_spec(),
        molecule_uri="molecule://x",
        molecule_hash="0" * 64,
        priority=priority,
    )


def _queued(name: str = "job", *, priority: int = 100) -> Job:
    job = _job(name, priority=priority)
    job.transition_to(JobStatus.QUEUED, actor="test")
    return job


# --------------------------------------------------------------------------- #
# Паритет двух реализаций
# --------------------------------------------------------------------------- #
def test_job_round_trip_is_lossless(store: JobStore) -> None:
    job = _queued("круговой")
    store.save(job)
    loaded = store.load(job.id)
    assert loaded == job
    assert store.exists(job.id)
    assert len(store) == 1
    with pytest.raises(LookupError):
        store.load("нет-такого")
    assert not store.exists("нет-такого")


def test_list_filters_by_status_and_sorts_fresh_first(store: JobStore) -> None:
    first, second, third = _queued("a"), _queued("b"), _job("c")
    for job in (first, second, third):
        store.save(job)
        time.sleep(0.01)
    assert {job.id for job in store.list()} == {first.id, second.id, third.id}
    queued = store.list(JobStatus.QUEUED)
    assert {job.id for job in queued} == {first.id, second.id}
    stamps = [job.created_at for job in store.list()]
    assert stamps == sorted(stamps, reverse=True)


def test_update_persists_and_a_failed_update_does_not(store: JobStore) -> None:
    job = _queued()
    store.save(job)
    updated = store.update(job.id, lambda item: item.transition_to(JobStatus.STARTING))
    assert updated.status is JobStatus.STARTING
    assert store.load(job.id).status is JobStatus.STARTING

    def illegal(item: Job) -> None:
        item.tags = ("не-должно-сохраниться",)
        item.transition_to(JobStatus.COMPLETED)  # из STARTING напрямую нельзя

    with pytest.raises(InvalidJobTransitionError):
        store.update(job.id, illegal)
    reloaded = store.load(job.id)
    assert reloaded.status is JobStatus.STARTING
    assert reloaded.tags == ()
    with pytest.raises(LookupError):
        store.update("нет-такого", lambda _item: None)


def test_structure_result_and_geometry_round_trip(store: JobStore) -> None:
    job = _job()
    store.save(job)
    assert store.load_result(job.id) is None
    assert store.load_geometry(job.id) is None
    store.store_molecule(job.id, HYDROGEN)
    assert store.load_molecule(job.id) == HYDROGEN
    payload = json.dumps({"energy": -1.1, "название": "водород"}, ensure_ascii=False)
    store.save_result(job.id, payload)
    assert store.load_result(job.id) == payload
    assert store.result_locator(job.id)
    store.save_geometry(job.id, HYDROGEN)
    assert store.load_geometry(job.id) == HYDROGEN
    # Повторная запись заменяет, а не дублирует.
    store.save_result(job.id, "{}")
    assert store.load_result(job.id) == "{}"
    assert store.molecule_locator(job.id)


def test_checkpoint_is_per_attempt_and_checked_against_its_digest(store: JobStore) -> None:
    job = _job()
    store.save(job)
    assert store.load_checkpoint(job.id, 0) is None
    reference = store.save_checkpoint(job.id, 0, '{"a": 1}')
    other = store.save_checkpoint(job.id, 1, '{"a": 2}')
    assert store.load_checkpoint(job.id, 0, reference.uri) == '{"a": 1}'
    assert store.load_checkpoint(job.id, 1, other.uri) == '{"a": 2}'
    # Сумма от другого содержимого — подмена: читать нельзя.
    with pytest.raises(JobCheckpointInvalidError):
        store.load_checkpoint(job.id, 0, other.uri)
    assert reference.sha256 != other.sha256


@pytest.fixture(params=["local", "postgres"])
def catalog(request: pytest.FixtureRequest, tmp_path: Path) -> LocalCatalog | PostgresCatalog:
    if request.param == "local":
        return LocalCatalog(tmp_path / "catalog")
    return PostgresCatalog(request.getfixturevalue("database"))


def test_catalog_projects_and_molecules(catalog: LocalCatalog | PostgresCatalog) -> None:
    beta = catalog.create_project("б-проект")
    alpha = catalog.create_project("а-проект")
    assert [p.name for p in catalog.list_projects()] == ["а-проект", "б-проект"]
    assert catalog.get_project(alpha.id).name == "а-проект"
    record = catalog.create_molecule(project_id=alpha.id, name="водород", content=HYDROGEN)
    assert catalog.get_molecule(record.id).molecule == Molecule.from_xyz(HYDROGEN, name="водород")
    assert [m.id for m in catalog.list_molecules(alpha.id)] == [record.id]
    assert catalog.list_molecules(beta.id) == ()
    assert len(catalog) == 1
    with pytest.raises(CatalogEntryNotFoundError):
        catalog.get_project("нет")
    with pytest.raises(CatalogEntryNotFoundError):
        catalog.get_molecule("нет")
    with pytest.raises(CatalogEntryNotFoundError):
        catalog.create_molecule(project_id="нет", name="x", content=HYDROGEN)
    with pytest.raises(UnsupportedStructureFormatError):
        catalog.create_molecule(project_id=alpha.id, name="x", content=HYDROGEN, fmt="sdf")


# --------------------------------------------------------------------------- #
# Свойства PostgreSQL
# --------------------------------------------------------------------------- #
def test_migrations_are_applied_once_and_idempotent(fresh_database: str) -> None:
    database = PostgresDatabase(fresh_database, migrate=False)
    assert database.migrate() == tuple(version for version, _ in MIGRATIONS)
    assert database.migrate() == ()
    assert database.schema_version() == MIGRATIONS[-1][0]


def test_concurrent_start_applies_each_migration_exactly_once(fresh_database: str) -> None:
    errors: list[BaseException] = []

    def start() -> None:
        try:
            PostgresDatabase(fresh_database)  # создание применяет миграции
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=start) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    database = PostgresDatabase(fresh_database, migrate=False)
    with database.connect() as connection:
        rows = connection.execute("SELECT version FROM schema_migrations").fetchall()
    assert sorted(row[0] for row in rows) == [version for version, _ in MIGRATIONS]


def test_concurrent_updates_are_not_lost(database: PostgresDatabase) -> None:
    store = PostgresJobStore(database)
    job = _job(priority=0)
    store.save(job)
    workers, increments = 6, 8

    def work() -> None:
        for _ in range(increments):
            store.update(job.id, lambda item: setattr(item, "priority", item.priority + 1))

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert store.load(job.id).priority == workers * increments


def test_queue_hands_every_job_to_exactly_one_worker_in_priority_order(
    database: PostgresDatabase,
) -> None:
    store = PostgresJobStore(database)
    jobs = [
        _queued(f"j{i}", priority=priority) for i, priority in enumerate([5, 90, 40, 70, 10, 60])
    ]
    for job in jobs:
        store.save(job)
        time.sleep(0.005)

    # Последовательно: порядок — по убыванию приоритета.
    sequential = []
    while (claimed := store.claim_next_queued(actor="w")) is not None:
        sequential.append(claimed.priority)
        assert claimed.status is JobStatus.STARTING
        store.update(claimed.id, lambda item: item.transition_to(JobStatus.RUNNING))
    assert sequential == sorted(sequential, reverse=True)
    assert store.claim_next_queued(actor="w") is None

    # Параллельно: каждое задание достаётся ровно одному воркеру.
    many = [_queued(f"p{i}") for i in range(24)]
    for job in many:
        store.save(job)
    taken: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            claimed = store.claim_next_queued(actor="w")
            if claimed is None:
                return
            with lock:
                taken.append(claimed.id)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(taken) == sorted(job.id for job in many)
    assert len(set(taken)) == len(taken)


def test_a_corrupted_row_is_refused_not_silently_read(database: PostgresDatabase) -> None:
    store = PostgresJobStore(database)
    job = _job()
    store.save(job)
    with database.connect() as connection:
        connection.execute("UPDATE jobs SET body = %s WHERE id = %s", ('{"id": 1}', job.id))
    with pytest.raises(ValidationError):
        store.load(job.id)


def test_dsn_is_redacted_for_messages() -> None:
    assert "secret" not in _redact("postgresql://user:secret@db.example:5432/ql")
    assert _redact("postgresql://user:secret@db.example:5432/ql").startswith("postgresql://user@")
    assert "secret" not in _redact("host=db user=u password=secret dbname=ql")


# --------------------------------------------------------------------------- #
# Сквозные сценарии: воркер, сервер, CLI, интерфейс
# --------------------------------------------------------------------------- #
def _submit_hydrogen(client: TestClient) -> str:
    project = client.post("/projects", json={"name": "проект"}).json()
    molecule = client.post(
        f"/projects/{project['id']}/molecules", json={"name": "H2", "content": HYDROGEN}
    ).json()
    spec = client.post(
        "/calculations/plan",
        json={"task": "single_point", "profile": "screening", "moleculeId": molecule["id"]},
    ).json()["spec"]
    accepted = client.post("/jobs", json={"moleculeId": molecule["id"], "spec": spec})
    assert accepted.status_code == 202
    return str(accepted.json()["id"])


def test_server_on_postgres_runs_a_job_and_keeps_its_checkpoint(
    fresh_database: str, tmp_path: Path
) -> None:
    app = create_app(tmp_path / "data", fresh_database)
    services: Services = app.state.services
    assert isinstance(services.jobs, PostgresJobStore)
    assert isinstance(services.catalog, PostgresCatalog)
    with TestClient(app) as client:
        assert client.get("/ready").json()["status"] == "ready"
        job_id = _submit_hydrogen(client)
        outcomes = run_pending_jobs(services.jobs, services.catalog)
        assert [(o.job_id, o.status) for o in outcomes] == [
            (job_id, JobStatus.COMPLETED_WITH_WARNINGS)
        ]
        result = client.get(f"/jobs/{job_id}/result").json()
        assert result["energyHartree"] < -1.0
        assert client.get("/jobs").json()["total"] == 1
    job = services.jobs.load(job_id)
    assert job.result_uri is not None
    assert job.result_uri.startswith("postgres://")
    # Воркер теперь пишет контрольные точки (раньше — только CLI).
    assert job.checkpoint_uri is not None
    assert services.jobs.load_checkpoint(job.id, job.attempt, job.checkpoint_uri)


def test_not_ready_when_the_database_is_unreachable(fresh_database: str, tmp_path: Path) -> None:
    app = create_app(tmp_path / "data", fresh_database)
    services: Services = app.state.services
    assert isinstance(services.jobs, PostgresJobStore)
    services.jobs.database.dsn = fresh_database.replace("/ql_", "/нет_такой_")
    with TestClient(app) as client:
        assert client.get("/ready").json()["status"] == "not_ready"


def test_embedded_worker_completes_a_job_through_the_api(
    fresh_database: str, tmp_path: Path
) -> None:
    app = create_app(tmp_path / "data", fresh_database, embedded_worker=True)
    with TestClient(app) as client:
        job_id = _submit_hydrogen(client)
        deadline = time.monotonic() + 60
        status = ""
        while time.monotonic() < deadline:
            status = client.get(f"/jobs/{job_id}").json()["status"]
            if status in {"completed", "completed_with_warnings", "failed"}:
                break
            time.sleep(0.2)
        assert status in {"completed", "completed_with_warnings"}
        assert client.get(f"/jobs/{job_id}/result").status_code == 200


def test_cli_run_and_job_list_on_postgres(
    fresh_database: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    water = str(FIXTURES / "water.xyz")
    code = main(
        [
            "--lang", "ru", "--data-dir", str(tmp_path), "--database-url", fresh_database,
            "run", water, "--task", "energy", "--method", "hf", "--basis", "sto-3g",
        ]
    )  # fmt: skip
    assert code == 0
    capsys.readouterr()
    store = PostgresJobStore(PostgresDatabase(fresh_database))
    (job,) = store.list()
    assert job.status in {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}
    result = json.loads(store.load_result(job.id) or "{}")
    assert result["energy_hartree"] == pytest.approx(-74.9630296640, abs=1e-6)
    # Ни одного файла хранилища в каталоге данных: всё в базе.
    assert not list(Path(tmp_path).rglob("*.json"))
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "--database-url",
            fresh_database,
            "job",
            "list",
        ]
    )
    assert code == 0
    assert job.id[:8] in capsys.readouterr().out


def test_database_url_comes_from_the_environment(
    fresh_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUANTUMLAB_DATABASE_URL", fresh_database)
    services = Services(tmp_path / "data")
    assert isinstance(services.jobs, PostgresJobStore)
    monkeypatch.delenv("QUANTUMLAB_DATABASE_URL")
    assert isinstance(Services(tmp_path / "data").jobs, LocalJobStore)


def test_web_ui_is_served_and_the_api_answers_under_its_contract_prefix(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "data")) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code in {302, 307}
        page = client.get("/ui/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert 'id="app"' in page.text
        assert client.get("/ui/app.js").status_code == 200
        assert client.get("/ui/style.css").status_code == 200
        # Контракт объявляет сервер /api/v1: тот же API отвечает и по префиксу.
        assert client.get("/api/v1/health").json() == {"status": "ok"}
        assert client.get("/api/v1/capabilities").status_code == 200
        assert client.get("/api/v1/i18n/ru").json()["gui.title"]
        assert client.get("/api/v1/i18n/en").json()["gui.title"]
