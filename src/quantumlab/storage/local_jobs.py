"""Файловое хранилище заданий для локального режима.

Требования, которые реализованы здесь и обязательны для любой будущей реализации
(PostgreSQL и др.):

1. **Атомарность**: файл задания перезаписывается через временный файл и
   ``Path.replace`` — процесс, упавший посреди записи, не оставляет «половину»
   задания (§14 ТЗ).
2. **Валидация на чтение**: сохранённый JSON разбирается строгой моделью
   :class:`~quantumlab.domain.job.Job`, поэтому повреждённое задание не
   приводит к неопределённому поведению.
3. **Идемпотентность повторов**: номер попытки хранится вместе с заданием и
   входит в имя чекпоинта.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from quantumlab.domain.job import Job
from quantumlab.domain.result import ArtifactKind, ArtifactRef
from quantumlab.engine.checkpoint import (
    CHECKPOINT_ARTIFACT_SCHEMA,
    checkpoint_uri,
    payload_sha256,
    sha256_from_uri,
)
from quantumlab.errors import JobCheckpointInvalidError
from quantumlab.jobs.state_machine import JobStatus


class LocalJobStore:
    """Хранилище заданий в виде каталога JSON-файлов.

    Структура каталога::

        <root>/
          jobs/<job_id>.json
          molecules/<job_id>.xyz
          checkpoints/<job_id>-<attempt>.json
    """

    def __init__(self, root: Path) -> None:
        """Создаёт хранилище (и каталоги) в ``root``."""
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.molecules_dir = self.root / "molecules"
        self.checkpoints_dir = self.root / "checkpoints"
        self.results_dir = self.root / "results"
        self.geometries_dir = self.root / "geometries"
        for directory in (
            self.jobs_dir,
            self.molecules_dir,
            self.checkpoints_dir,
            self.results_dir,
            self.geometries_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- пути ---------------------------------------------------------------- #
    def job_path(self, job_id: str) -> Path:
        """Путь к файлу задания."""
        return self.jobs_dir / f"{job_id}.json"

    def molecule_path(self, job_id: str) -> Path:
        """Путь к сохранённой структуре задания."""
        return self.molecules_dir / f"{job_id}.xyz"

    def checkpoint_path(self, job_id: str, attempt: int) -> Path:
        """Путь чекпоинта: зависит от попытки, поэтому повторы не затирают его."""
        return self.checkpoints_dir / f"{job_id}-{attempt}.json"

    # -- операции ------------------------------------------------------------ #
    def save(self, job: Job) -> Path:
        """Атомарно сохраняет задание."""
        payload = json.dumps(
            job.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
        )
        target = self.job_path(job.id)
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
        return target

    def result_path(self, job_id: str) -> Path:
        """Путь к JSON-результату расчёта.

        Результат хранится отдельным файлом, а не внутри метаданных задания:
        он на порядки больше и нужен не каждому читателю списка заданий.
        """
        return self.results_dir / f"{job_id}.json"

    def geometry_path(self, job_id: str) -> Path:
        """Путь к итоговой геометрии расчёта, меняющего структуру."""
        return self.geometries_dir / f"{job_id}.xyz"

    def save_geometry(self, job_id: str, xyz_text: str) -> Path:
        """Атомарно сохраняет итоговую геометрию (например, после оптимизации)."""
        target = self.geometry_path(job_id)
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(xyz_text, encoding="utf-8")
        temporary.replace(target)
        return target

    def save_result(self, job_id: str, payload: str) -> Path:
        """Атомарно сохраняет результат расчёта.

        Атомарность та же, что и у метаданных задания: сначала временный файл,
        затем ``rename``. Читатель никогда не видит наполовину записанный
        результат — иначе упавший процесс оставил бы битый артефакт.
        """
        target = self.result_path(job_id)
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
        return target

    def save_checkpoint(self, job_id: str, attempt: int, payload: str) -> ArtifactRef:
        """Атомарно сохраняет контрольную точку и возвращает ссылку на артефакт.

        Атомарность та же, что у результата: временный файл и ``rename``. Это
        принципиально именно здесь — контрольную точку пишут посреди расчёта,
        то есть ровно тогда, когда процесс чаще всего и падает.

        Путь зависит от попытки, поэтому повтор не затирает состояние
        предыдущего запуска: при разборе падения нужно видеть оба.
        """
        target = self.checkpoint_path(job_id, attempt)
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
        digest = payload_sha256(payload)
        return ArtifactRef(
            kind=ArtifactKind.CHECKPOINT,
            uri=checkpoint_uri(target.name, digest),
            sha256=digest,
            size_bytes=len(payload.encode("utf-8")),
            schema_version=CHECKPOINT_ARTIFACT_SCHEMA,
        )

    def load_checkpoint(self, job_id: str, attempt: int, uri: str | None = None) -> str | None:
        """Возвращает содержимое контрольной точки либо ``None``, если её нет.

        Если передан ``uri``, контрольная сумма из него сверяется с содержимым.
        Несовпадение — ошибка, а не повод считать с тем, что лежит на диске:
        подменённый или обрезанный файл дал бы расчёт, относящийся к другой
        задаче. Отсутствие файла — нормальная ситуация: задание могло не
        дожить до записи.
        """
        path = self.checkpoint_path(job_id, attempt)
        if not path.exists():
            return None
        payload = path.read_text(encoding="utf-8")
        expected = sha256_from_uri(uri) if uri is not None else None
        if expected is not None and payload_sha256(payload) != expected:
            msg = (
                f"Контрольная сумма файла {path.name} не совпадает с сохранённой "
                "в ссылке на артефакт. Файл изменён после записи."
            )
            raise JobCheckpointInvalidError(msg)
        return payload

    def store_molecule(self, job_id: str, xyz_text: str) -> Path:
        """Сохраняет структуру задания как XYZ-файл."""
        path = self.molecule_path(job_id)
        path.write_text(xyz_text, encoding="utf-8")
        return path

    def load(self, job_id: str) -> Job:
        """Читает задание; при отсутствии файла бросает ``LookupError``."""
        path = self.job_path(job_id)
        if not path.exists():
            raise LookupError(job_id)
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        return Job.model_validate(data)

    def exists(self, job_id: str) -> bool:
        """Существует ли задание."""
        return self.job_path(job_id).exists()

    def list(self, status: JobStatus | None = None) -> tuple[Job, ...]:
        """Все задания, опционально отфильтрованные по статусу.

        Сортировка — по дате создания, свежие сверху: это порядок, в котором
        список показывает GUI.
        """
        jobs = [
            Job.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in self.jobs_dir.glob("*.json")
        ]
        if status is not None:
            jobs = [job for job in jobs if job.status is status]
        return tuple(sorted(jobs, key=lambda job: job.created_at, reverse=True))

    def update(self, job_id: str, mutate: Callable[[Job], None]) -> Job:
        """Читает задание, применяет изменение и атомарно сохраняет.

        Валидация перехода состояния выполняется внутри ``mutate``
        (см. :meth:`Job.transition_to`), поэтому недопустимый переход не
        записывается в хранилище.
        """
        job = self.load(job_id)
        mutate(job)
        self.save(job)
        return job

    def __len__(self) -> int:
        """Число сохранённых заданий."""
        return sum(1 for _ in self.jobs_dir.glob("*.json"))
