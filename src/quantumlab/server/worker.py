"""Исполнитель очереди заданий.

``POST /jobs`` только принимает расчёт и возвращает ``202``: выполнять
многосекундный расчёт внутри HTTP-запроса нельзя ни по таймаутам, ни по
восстановлению после сбоя (§14 ТЗ). Выполнение — отдельный шаг, который
в развёртывании живёт своим процессом, а в разработке и в тестах вызывается
явно через :func:`run_pending_jobs`.

Логика намеренно та же, что в CLI: один и тот же домен, одно и то же ядро,
один и тот же способ записывать диагноз. Расхождение между CLI и API в этом
месте означало бы два разных поведения для одного расчёта.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import partial
from typing import TypeGuard

from quantumlab.domain.job import Job
from quantumlab.domain.molecule import Molecule
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.errors import CatalogEntryNotFoundError, QuantumLabError
from quantumlab.jobs.state_machine import JobStatus
from quantumlab.storage.base import Catalog, ClaimingJobStore, JobStore


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """Итог обработки одного задания."""

    job_id: str
    status: JobStatus
    error_code: str | None = None


def _resolve_molecule(catalog: Catalog, uri: str) -> Molecule:
    """Достаёт структуру по ``molecule://<id>``.

    URI хранится в задании, а не структура целиком: иначе повтор задания
    мог бы незаметно пойти по другой геометрии.
    """
    prefix = "molecule://"
    if not uri.startswith(prefix):
        raise CatalogEntryNotFoundError("molecule", uri)
    return catalog.get_molecule(uri[len(prefix) :]).molecule


def _transition(job: Job, *, status: JobStatus) -> None:
    """Переводит задание в новый статус."""
    job.transition_to(status, actor="worker")


def _mark_failed(job: Job, *, code: str, params: dict[str, str]) -> None:
    """Переводит задание в «не выполнено», сохраняя машиночитаемый диагноз."""
    job.error_code = code
    job.error_params = params
    job.transition_to(JobStatus.FAILED, actor="worker")


def _mark_finished(job: Job, *, result_uri: str, status: JobStatus) -> None:
    """Привязывает результат к заданию и закрывает его."""
    job.result_uri = result_uri
    job.transition_to(status, actor="worker")


def run_pending_jobs(
    jobs: JobStore,
    catalog: Catalog,
    *,
    engine: ReferenceEngine | None = None,
    limit: int | None = None,
) -> tuple[WorkerOutcome, ...]:
    """Выполняет задания из очереди по одному.

    Возвращает итог по каждому обработанному заданию. Ошибка в одном задании
    не останавливает остальные: диагноз пишется в само задание, а не в процесс.

    Если хранилище умеет атомарную выдачу (PostgreSQL, ``claim_next_queued``),
    задания берутся по одному под блокировкой ``SKIP LOCKED``: несколько
    воркеров могут работать с одной очередью, не дублируя расчёты. Файловое
    хранилище рассчитано на одного воркера.
    """
    core = engine or ReferenceEngine()
    outcomes: list[WorkerOutcome] = []
    if _can_claim(jobs):
        while limit is None or len(outcomes) < limit:
            claimed = jobs.claim_next_queued(actor="worker")
            if claimed is None:
                break
            outcomes.append(_run_one(jobs, catalog, core, claimed, started=True))
        return tuple(outcomes)
    queue = list(jobs.list(JobStatus.QUEUED))
    queue.sort(key=lambda item: (-item.priority, item.created_at))
    for job in queue[:limit] if limit is not None else queue:
        outcomes.append(_run_one(jobs, catalog, core, job))
    return tuple(outcomes)


def _can_claim(jobs: JobStore) -> TypeGuard[ClaimingJobStore]:
    return callable(getattr(jobs, "claim_next_queued", None))


def serve_forever(
    jobs: JobStore,
    catalog: Catalog,
    stop: threading.Event,
    *,
    poll_seconds: float = 1.0,
    engine: ReferenceEngine | None = None,
) -> None:
    """Цикл воркера: обрабатывает очередь, пока не выставлен ``stop``.

    Ошибка хранилища (например, обрыв соединения) не убивает цикл: она
    пропускается, и опрос повторяется — воркер переживает перезапуск базы.
    """
    core = engine or ReferenceEngine()
    while not stop.is_set():
        try:
            handled = run_pending_jobs(jobs, catalog, engine=core)
        except Exception:
            handled = ()
        if not handled:
            stop.wait(poll_seconds)


def _run_one(
    jobs: JobStore,
    catalog: Catalog,
    engine: ReferenceEngine,
    job: Job,
    *,
    started: bool = False,
) -> WorkerOutcome:
    """Выполняет одно задание и возвращает итог.

    ``started`` — задание уже переведено в ``STARTING`` (его выдало хранилище).
    """
    # Машина состояний ведёт задание через STARTING: QUEUED → STARTING → RUNNING.
    # Пропускать промежуточный статус нельзя — иначе потерялось бы различие
    # между «не смог стартовать» и «упал в работе» (§14 ТЗ).
    if not started:
        jobs.update(job.id, partial(_transition, status=JobStatus.STARTING))
    try:
        molecule = _resolve_molecule(catalog, job.molecule_uri)
    except QuantumLabError as error:
        params = {key: str(value) for key, value in error.params.items()}
        jobs.update(job.id, partial(_mark_failed, code=str(error.code), params=params))
        return WorkerOutcome(job.id, JobStatus.FAILED, str(error.code))

    jobs.update(job.id, partial(_transition, status=JobStatus.RUNNING))

    def persist_checkpoint(payload: str) -> None:
        reference = jobs.save_checkpoint(job.id, job.attempt, payload)
        jobs.update(job.id, partial(_set_checkpoint, uri=reference.uri))

    try:
        stored = (
            jobs.load_checkpoint(job.id, job.attempt, job.checkpoint_uri)
            if job.checkpoint_uri is not None
            else None
        )
        result = engine.run(
            EngineRequest(
                job_id=job.id,
                molecule=molecule,
                spec=job.spec,
                threads=job.spec.resources.threads or 1,
                checkpoint=stored,
            ),
            checkpoint_sink=persist_checkpoint,
        )
    except QuantumLabError as error:
        code = str(error.code)
        params = {key: str(value) for key, value in error.params.items()}
        jobs.update(job.id, partial(_mark_failed, code=code, params=params))
        return WorkerOutcome(job.id, JobStatus.FAILED, code)

    jobs.save_result(job.id, result.model_dump_json(indent=2))
    result_uri = jobs.result_locator(job.id)
    if result.final_molecule is not None:
        jobs.save_geometry(job.id, result.final_molecule.to_xyz())
    final = JobStatus.COMPLETED_WITH_WARNINGS if result.warnings else JobStatus.COMPLETED
    jobs.update(job.id, partial(_mark_finished, result_uri=result_uri, status=final))
    return WorkerOutcome(job.id, final)


def _set_checkpoint(job: Job, *, uri: str) -> None:
    """Запоминает ссылку на последнюю контрольную точку."""
    job.checkpoint_uri = uri
