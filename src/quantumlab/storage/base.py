"""Контракты хранилищ: общий интерфейс для файлового режима и PostgreSQL.

Потребители (CLI, REST-сервер, воркер) знают только эти протоколы. Различие
реализаций — в способе хранения и в **гарантиях конкурентности**:

* файловое хранилище рассчитано на один процесс (чтение–изменение–запись без
  блокировок);
* PostgreSQL изменяет задание в транзакции под ``SELECT … FOR UPDATE`` и выдаёт
  задания воркерам через ``FOR UPDATE SKIP LOCKED`` — два воркера никогда не
  получат одно и то же задание.

Ссылки на данные — «локаторы» (строки). Для файлов это ``file://…``, для базы —
``postgres://<таблица>/<id>``. Потребитель не должен разбирать локатор: для
чтения данных есть ``load_*``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from quantumlab.domain.job import Job
from quantumlab.domain.result import ArtifactRef
from quantumlab.jobs.state_machine import JobStatus
from quantumlab.storage.local_catalog import MoleculeRecord, ProjectRecord


class JobStore(Protocol):
    """Хранилище заданий и связанных с ними данных."""

    def describe(self) -> str:
        """Человекочитаемое описание места хранения (для сообщений об ошибках)."""
        ...

    def save(self, job: Job) -> object:
        """Сохраняет задание (создаёт или заменяет)."""
        ...

    def load(self, job_id: str) -> Job:
        """Читает задание; при отсутствии бросает ``LookupError``."""
        ...

    def exists(self, job_id: str) -> bool:
        """Существует ли задание."""
        ...

    def list(self, status: JobStatus | None = None) -> tuple[Job, ...]:
        """Задания (опционально по статусу), свежие сверху."""
        ...

    def update(self, job_id: str, mutate: Callable[[Job], None]) -> Job:
        """Читает, изменяет и сохраняет задание; недопустимый переход не записывается."""
        ...

    def __len__(self) -> int:
        """Число заданий."""
        ...

    # -- структура задания ----------------------------------------------------
    def store_molecule(self, job_id: str, xyz_text: str) -> object:
        """Сохраняет структуру задания."""
        ...

    def molecule_locator(self, job_id: str) -> str:
        """Локатор структуры задания."""
        ...

    def load_molecule(self, job_id: str) -> str:
        """Текст XYZ структуры задания."""
        ...

    # -- результат и геометрия -----------------------------------------------
    def save_result(self, job_id: str, payload: str) -> object:
        """Сохраняет JSON результата."""
        ...

    def result_locator(self, job_id: str) -> str:
        """Локатор результата."""
        ...

    def load_result(self, job_id: str) -> str | None:
        """JSON результата или ``None``, если расчёт ещё не дал результата."""
        ...

    def save_geometry(self, job_id: str, xyz_text: str) -> object:
        """Сохраняет итоговую геометрию."""
        ...

    def load_geometry(self, job_id: str) -> str | None:
        """Итоговая геометрия (XYZ) или ``None``."""
        ...

    # -- контрольные точки -----------------------------------------------------
    def save_checkpoint(self, job_id: str, attempt: int, payload: str) -> ArtifactRef:
        """Сохраняет контрольную точку и возвращает ссылку на артефакт."""
        ...

    def load_checkpoint(self, job_id: str, attempt: int, uri: str | None = None) -> str | None:
        """Контрольная точка (с проверкой суммы из ``uri``) или ``None``."""
        ...


class ClaimingJobStore(JobStore, Protocol):
    """Хранилище, умеющее атомарно выдавать задания нескольким воркерам."""

    def claim_next_queued(self, *, actor: str) -> Job | None:
        """Берёт самое приоритетное задание очереди и переводит его в ``STARTING``."""
        ...


class Catalog(Protocol):
    """Каталог проектов и структур."""

    def create_project(self, name: str) -> ProjectRecord:
        """Создаёт проект."""
        ...

    def get_project(self, project_id: str) -> ProjectRecord:
        """Проект или ``CatalogEntryNotFoundError``."""
        ...

    def list_projects(self) -> tuple[ProjectRecord, ...]:
        """Проекты по имени."""
        ...

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
        """Разбирает и сохраняет структуру."""
        ...

    def get_molecule(self, molecule_id: str) -> MoleculeRecord:
        """Структура или ``CatalogEntryNotFoundError``."""
        ...

    def list_molecules(self, project_id: str) -> tuple[MoleculeRecord, ...]:
        """Структуры проекта по имени."""
        ...

    def __len__(self) -> int:
        """Число структур."""
        ...
