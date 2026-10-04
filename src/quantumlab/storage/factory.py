"""Выбор хранилища: файлы (по умолчанию) или PostgreSQL по DSN.

DSN задаётся флагом ``--database-url`` либо переменной окружения
``QUANTUMLAB_DATABASE_URL``. Пустое значение означает файловый режим.
"""

from __future__ import annotations

import os
from pathlib import Path

from quantumlab.storage.base import Catalog, JobStore
from quantumlab.storage.local_catalog import LocalCatalog
from quantumlab.storage.local_jobs import LocalJobStore

DATABASE_URL_ENV = "QUANTUMLAB_DATABASE_URL"


def resolve_database_url(explicit: str | None = None) -> str | None:
    """DSN из аргумента или окружения; ``None`` — файловое хранилище."""
    value = explicit if explicit else os.environ.get(DATABASE_URL_ENV, "")
    return value.strip() or None


def open_job_store(data_dir: Path | str, database_url: str | None = None) -> JobStore:
    """Хранилище заданий: PostgreSQL при наличии DSN, иначе каталог ``data_dir``."""
    dsn = resolve_database_url(database_url)
    if dsn is None:
        return LocalJobStore(Path(data_dir))
    from quantumlab.storage.postgres import PostgresDatabase, PostgresJobStore

    return PostgresJobStore(PostgresDatabase(dsn))


def open_catalog(data_dir: Path | str, database_url: str | None = None) -> Catalog:
    """Каталог проектов и структур: PostgreSQL при наличии DSN, иначе файлы."""
    dsn = resolve_database_url(database_url)
    if dsn is None:
        return LocalCatalog(Path(data_dir) / "catalog")
    from quantumlab.storage.postgres import PostgresCatalog, PostgresDatabase

    return PostgresCatalog(PostgresDatabase(dsn))
