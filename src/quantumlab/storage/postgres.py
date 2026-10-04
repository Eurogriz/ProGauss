"""Хранилище заданий и каталога в PostgreSQL.

Реализует те же протоколы, что и файловые хранилища
(:mod:`quantumlab.storage.base`), и добавляет то, чего файлы дать не могут:

* **транзакционное изменение задания** — ``update`` читает строку под
  ``SELECT … FOR UPDATE``, применяет изменение и записывает результат в одной
  транзакции. Два одновременных ``update`` одного задания не теряют друг друга;
  недопустимый переход состояния откатывает транзакцию;
* **очередь без дублей** — ``claim_next_queued`` выбирает задание
  ``FOR UPDATE SKIP LOCKED`` и в той же транзакции переводит его в ``STARTING``:
  параллельные воркеры никогда не получают одно задание дважды;
* **миграции схемы** — таблица ``schema_migrations`` и advisory-блокировка:
  параллельный старт нескольких процессов не применит миграцию дважды.

Форма данных. Полная доменная модель (``Job``, ``MoleculeRecord``) хранится в
``JSONB``-колонке ``body`` и **валидируется при чтении** той же строгой моделью,
что и в файловом режиме. Колонки ``status``, ``priority``, ``created_at``,
``project_id`` дублируют поля для индексов и фильтров. Дублирование безопасно:
обе записи делаются одним оператором, а источник истины — ``body``.

Соединения короткие (одно на операцию): пул в этом срезе не нужен, а ошибка
«соединение протухло» невозможна. Для высокой нагрузки пул — следующий шаг.

Требуется ``psycopg`` 3 (``pip install "quantumlab[postgres]"``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any

from quantumlab.domain.job import Job
from quantumlab.domain.molecule import Molecule
from quantumlab.domain.result import ArtifactKind, ArtifactRef
from quantumlab.engine.checkpoint import (
    CHECKPOINT_ARTIFACT_SCHEMA,
    checkpoint_uri,
    payload_sha256,
    sha256_from_uri,
)
from quantumlab.errors import (
    CatalogEntryNotFoundError,
    JobCheckpointInvalidError,
    UnsupportedStructureFormatError,
)
from quantumlab.jobs.state_machine import JobStatus
from quantumlab.storage.local_catalog import PARSABLE_FORMATS, MoleculeRecord, ProjectRecord

if TYPE_CHECKING:
    import psycopg

#: Ключ advisory-блокировки миграций (произвольная константа проекта).
_MIGRATION_LOCK = 7_262_024

#: Миграции схемы. Номер — порядок применения; применённые не переписываются:
#: изменения схемы добавляются новыми элементами.
MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE jobs (
            id          TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL,
            status      TEXT NOT NULL,
            priority    INTEGER NOT NULL,
            attempt     INTEGER NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            body        JSONB NOT NULL
        );
        CREATE INDEX jobs_queue_idx ON jobs (status, priority DESC, created_at);
        CREATE INDEX jobs_project_idx ON jobs (project_id);

        CREATE TABLE job_molecules (
            job_id  TEXT PRIMARY KEY,
            xyz     TEXT NOT NULL
        );
        CREATE TABLE job_results (
            job_id   TEXT PRIMARY KEY,
            payload  TEXT NOT NULL,
            saved_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE job_geometries (
            job_id  TEXT PRIMARY KEY,
            xyz     TEXT NOT NULL
        );
        CREATE TABLE job_checkpoints (
            job_id    TEXT NOT NULL,
            attempt   INTEGER NOT NULL,
            payload   TEXT NOT NULL,
            sha256    TEXT NOT NULL,
            saved_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (job_id, attempt)
        );

        CREATE TABLE projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL,
            body        JSONB NOT NULL
        );
        CREATE TABLE molecules (
            id          TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
            name        TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL,
            body        JSONB NOT NULL
        );
        CREATE INDEX molecules_project_idx ON molecules (project_id, name);
        """,
    ),
)


def _import_psycopg() -> ModuleType:
    try:
        import psycopg
    except ImportError as error:  # pragma: no cover - зависит от окружения
        msg = (
            "Для PostgreSQL нужен пакет psycopg 3: "
            'pip install "quantumlab[postgres]" (или psycopg[binary])'
        )
        raise RuntimeError(msg) from error
    return psycopg


class PostgresDatabase:
    """Подключение к базе и применение миграций."""

    def __init__(self, dsn: str, *, migrate: bool = True) -> None:
        """Запоминает DSN и (по умолчанию) применяет недостающие миграции."""
        self.dsn = dsn
        self._psycopg = _import_psycopg()
        if migrate:
            self.migrate()

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection[Any]]:
        """Короткая транзакционная сессия: ``commit`` при выходе, ``rollback`` при ошибке."""
        with self._psycopg.connect(self.dsn) as connection:
            yield connection

    def migrate(self) -> tuple[int, ...]:
        """Применяет недостающие миграции; возвращает номера применённых."""
        applied: list[int] = []
        with self.connect() as connection:
            # Блокировка до конца транзакции: параллельный процесс дождётся
            # и увидит уже применённые миграции.
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            done = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            for version, script in MIGRATIONS:
                if version in done:
                    continue
                connection.execute(script)
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                )
                applied.append(version)
        return tuple(applied)

    def schema_version(self) -> int:
        """Наибольший применённый номер миграции (0 — схема не создана)."""
        with self.connect() as connection:
            row = connection.execute("SELECT max(version) FROM schema_migrations").fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def _json(value: object) -> object:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


class PostgresJobStore:
    """Задания, результаты, геометрии и контрольные точки в PostgreSQL."""

    def __init__(self, database: PostgresDatabase) -> None:
        """Создаёт хранилище поверх подключённой базы."""
        self.database = database

    def describe(self) -> str:
        """DSN без пароля."""
        return _redact(self.database.dsn)

    # -- задания ---------------------------------------------------------------
    def save(self, job: Job) -> str:
        """Создаёт или заменяет задание одним оператором."""
        with self.database.connect() as connection:
            self._upsert(connection, job)
        return f"postgres://jobs/{job.id}"

    @staticmethod
    def _upsert(connection: psycopg.Connection[Any], job: Job) -> None:
        connection.execute(
            """
            INSERT INTO jobs (id, project_id, status, priority, attempt, created_at, body)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                project_id = EXCLUDED.project_id, status = EXCLUDED.status,
                priority = EXCLUDED.priority, attempt = EXCLUDED.attempt,
                body = EXCLUDED.body, updated_at = now()
            """,
            (
                job.id,
                job.project_id,
                job.status.value,
                job.priority,
                job.attempt,
                job.created_at,
                _json(job.model_dump(mode="json")),
            ),
        )

    def load(self, job_id: str) -> Job:
        """Читает задание и валидирует строгой моделью; нет задания — ``LookupError``."""
        with self.database.connect() as connection:
            row = connection.execute("SELECT body FROM jobs WHERE id = %s", (job_id,)).fetchone()
        if row is None:
            raise LookupError(job_id)
        return Job.model_validate(row[0])

    def exists(self, job_id: str) -> bool:
        """Есть ли задание."""
        with self.database.connect() as connection:
            return (
                connection.execute("SELECT 1 FROM jobs WHERE id = %s", (job_id,)).fetchone()
                is not None
            )

    def list(self, status: JobStatus | None = None) -> tuple[Job, ...]:
        """Задания (опционально по статусу), свежие сверху."""
        with self.database.connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT body FROM jobs ORDER BY created_at DESC, id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT body FROM jobs WHERE status = %s ORDER BY created_at DESC, id",
                    (status.value,),
                ).fetchall()
        return tuple(Job.model_validate(row[0]) for row in rows)

    def update(self, job_id: str, mutate: Callable[[Job], None]) -> Job:
        """Транзакционное чтение–изменение–запись под ``FOR UPDATE``."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT body FROM jobs WHERE id = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError(job_id)
            job = Job.model_validate(row[0])
            mutate(job)  # исключение откатывает транзакцию: недопустимый переход не пишется
            self._upsert(connection, job)
        return job

    def claim_next_queued(self, *, actor: str) -> Job | None:
        """Атомарно выдаёт самое приоритетное задание очереди.

        ``SKIP LOCKED`` пропускает строки, которые уже захвачены другим воркером,
        поэтому один и тот же ``job`` никогда не достаётся двум процессам.
        """
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT body FROM jobs WHERE status = %s
                ORDER BY priority DESC, created_at ASC
                FOR UPDATE SKIP LOCKED LIMIT 1
                """,
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            job = Job.model_validate(row[0])
            job.transition_to(JobStatus.STARTING, actor=actor)
            self._upsert(connection, job)
        return job

    def __len__(self) -> int:
        """Число заданий."""
        with self.database.connect() as connection:
            row = connection.execute("SELECT count(*) FROM jobs").fetchone()
        return int(row[0]) if row else 0

    # -- структура, результат, геометрия --------------------------------------
    def _put(self, table: str, column: str, job_id: str, value: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                f"INSERT INTO {table} (job_id, {column}) VALUES (%s, %s) "
                f"ON CONFLICT (job_id) DO UPDATE SET {column} = EXCLUDED.{column}",
                (job_id, value),
            )

    def _get(self, table: str, column: str, job_id: str) -> str | None:
        with self.database.connect() as connection:
            row = connection.execute(
                f"SELECT {column} FROM {table} WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def store_molecule(self, job_id: str, xyz_text: str) -> str:
        """Сохраняет структуру задания."""
        self._put("job_molecules", "xyz", job_id, xyz_text)
        return self.molecule_locator(job_id)

    def molecule_locator(self, job_id: str) -> str:
        """Локатор структуры."""
        return f"postgres://job_molecules/{job_id}"

    def load_molecule(self, job_id: str) -> str:
        """Текст XYZ; нет записи — ``LookupError``."""
        text = self._get("job_molecules", "xyz", job_id)
        if text is None:
            raise LookupError(job_id)
        return text

    def save_result(self, job_id: str, payload: str) -> str:
        """Сохраняет JSON результата."""
        self._put("job_results", "payload", job_id, payload)
        return self.result_locator(job_id)

    def result_locator(self, job_id: str) -> str:
        """Локатор результата."""
        return f"postgres://job_results/{job_id}"

    def load_result(self, job_id: str) -> str | None:
        """JSON результата либо ``None``."""
        return self._get("job_results", "payload", job_id)

    def save_geometry(self, job_id: str, xyz_text: str) -> str:
        """Сохраняет итоговую геометрию."""
        self._put("job_geometries", "xyz", job_id, xyz_text)
        return f"postgres://job_geometries/{job_id}"

    def load_geometry(self, job_id: str) -> str | None:
        """Итоговая геометрия либо ``None``."""
        return self._get("job_geometries", "xyz", job_id)

    # -- контрольные точки ------------------------------------------------------
    def save_checkpoint(self, job_id: str, attempt: int, payload: str) -> ArtifactRef:
        """Сохраняет контрольную точку (по одной на попытку) и возвращает ссылку."""
        digest = payload_sha256(payload)
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO job_checkpoints (job_id, attempt, payload, sha256)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (job_id, attempt) DO UPDATE SET
                    payload = EXCLUDED.payload, sha256 = EXCLUDED.sha256, saved_at = now()
                """,
                (job_id, attempt, payload, digest),
            )
        return ArtifactRef(
            kind=ArtifactKind.CHECKPOINT,
            uri=checkpoint_uri(f"{job_id}-{attempt}", digest),
            sha256=digest,
            size_bytes=len(payload.encode("utf-8")),
            schema_version=CHECKPOINT_ARTIFACT_SCHEMA,
        )

    def load_checkpoint(self, job_id: str, attempt: int, uri: str | None = None) -> str | None:
        """Контрольная точка; сумма из ``uri`` сверяется с содержимым."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM job_checkpoints WHERE job_id = %s AND attempt = %s",
                (job_id, attempt),
            ).fetchone()
        if row is None:
            return None
        payload = str(row[0])
        expected = sha256_from_uri(uri) if uri is not None else None
        if expected is not None and payload_sha256(payload) != expected:
            msg = (
                f"Контрольная сумма контрольной точки {job_id}-{attempt} не совпадает "
                "с сохранённой в ссылке на артефакт: запись изменена после сохранения."
            )
            raise JobCheckpointInvalidError(msg)
        return payload


class PostgresCatalog:
    """Проекты и структуры в PostgreSQL."""

    def __init__(self, database: PostgresDatabase) -> None:
        """Создаёт каталог поверх подключённой базы."""
        self.database = database

    def create_project(self, name: str) -> ProjectRecord:
        """Создаёт проект."""
        record = ProjectRecord(id=str(uuid.uuid4()), name=name)
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO projects (id, name, created_at, body) VALUES (%s, %s, %s, %s)",
                (record.id, record.name, record.created_at, _json(record.model_dump(mode="json"))),
            )
        return record

    def get_project(self, project_id: str) -> ProjectRecord:
        """Проект или ``CatalogEntryNotFoundError``."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT body FROM projects WHERE id = %s", (project_id,)
            ).fetchone()
        if row is None:
            raise CatalogEntryNotFoundError("project", project_id)
        return ProjectRecord.model_validate(row[0])

    def list_projects(self) -> tuple[ProjectRecord, ...]:
        """Проекты по имени."""
        with self.database.connect() as connection:
            rows = connection.execute("SELECT body FROM projects ORDER BY name, id").fetchall()
        return tuple(ProjectRecord.model_validate(row[0]) for row in rows)

    def create_molecule(
        self,
        *,
        project_id: str,
        name: str | None,
        content: str,
        fmt: str = "xyz",
        charge: int = 0,
        multiplicity: int = 1,
    ) -> MoleculeRecord:
        """Разбирает структуру и сохраняет её; проект обязан существовать."""
        self.get_project(project_id)
        if fmt not in PARSABLE_FORMATS:
            raise UnsupportedStructureFormatError(name=fmt)
        molecule = Molecule.from_xyz(
            content, name=name or "molecule", charge=charge, multiplicity=multiplicity
        )
        record = MoleculeRecord(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=name or molecule.formula,
            format=fmt,
            charge=charge,
            multiplicity=multiplicity,
            molecule=molecule,
        )
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO molecules (id, project_id, name, created_at, body) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    record.id,
                    record.project_id,
                    record.name,
                    record.created_at,
                    _json(record.model_dump(mode="json")),
                ),
            )
        return record

    def get_molecule(self, molecule_id: str) -> MoleculeRecord:
        """Структура или ``CatalogEntryNotFoundError``."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT body FROM molecules WHERE id = %s", (molecule_id,)
            ).fetchone()
        if row is None:
            raise CatalogEntryNotFoundError("molecule", molecule_id)
        return MoleculeRecord.model_validate(row[0])

    def list_molecules(self, project_id: str) -> tuple[MoleculeRecord, ...]:
        """Структуры проекта по имени."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT body FROM molecules WHERE project_id = %s ORDER BY name, id",
                (project_id,),
            ).fetchall()
        return tuple(MoleculeRecord.model_validate(row[0]) for row in rows)

    def __len__(self) -> int:
        """Число структур."""
        with self.database.connect() as connection:
            row = connection.execute("SELECT count(*) FROM molecules").fetchone()
        return int(row[0]) if row else 0


def _redact(dsn: str) -> str:
    """DSN без пароля — для сообщений и журналов."""
    if "://" in dsn and "@" in dsn:
        scheme, rest = dsn.split("://", 1)
        credentials, host = rest.rsplit("@", 1)
        user = credentials.split(":", 1)[0]
        return f"{scheme}://{user}@{host}"
    return " ".join(part for part in dsn.split() if not part.lower().startswith("password="))
