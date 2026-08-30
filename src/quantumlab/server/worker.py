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

from dataclasses import dataclass
from functools import partial

from quantumlab.domain.job import Job
from quantumlab.domain.molecule import Molecule
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.errors import CatalogEntryNotFoundError, QuantumLabError
from quantumlab.jobs.state_machine import JobStatus
from quantumlab.storage.local_catalog import LocalCatalog
from quantumlab.storage.local_jobs import LocalJobStore


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """Итог обработки одного задания."""

    job_id: str
    status: JobStatus
    error_code: str | None = None


def _resolve_molecule(catalog: LocalCatalog, uri: str) -> Molecule:
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
    jobs: LocalJobStore,
    catalog: LocalCatalog,
    *,
    engine: ReferenceEngine | None = None,
    limit: int | None = None,
) -> tuple[WorkerOutcome, ...]:
    """Выполняет задания из очереди по одному.

    Возвращает итог по каждому обработанному заданию. Ошибка в одном задании
    не останавливает остальные: диагноз пишется в само задание, а не в процесс.
    """
    core = engine or ReferenceEngine()
    outcomes: list[WorkerOutcome] = []
    queue = list(jobs.list(JobStatus.QUEUED))
    queue.sort(key=lambda item: (-item.priority, item.created_at))
    for job in queue[:limit] if limit is not None else queue:
        outcomes.append(_run_one(jobs, catalog, core, job))
    return tuple(outcomes)


def _run_one(
    jobs: LocalJobStore, catalog: LocalCatalog, engine: ReferenceEngine, job: Job
) -> WorkerOutcome:
    """Выполняет одно задание и возвращает итог."""
    # Машина состояний ведёт задание через STARTING: QUEUED → STARTING → RUNNING.
    # Пропускать промежуточный статус нельзя — иначе потерялось бы различие
    # между «не смог стартовать» и «упал в работе» (§14 ТЗ).
    jobs.update(job.id, partial(_transition, status=JobStatus.STARTING))
    try:
        molecule = _resolve_molecule(catalog, job.molecule_uri)
    except QuantumLabError as error:
        params = {key: str(value) for key, value in error.params.items()}
        jobs.update(job.id, partial(_mark_failed, code=str(error.code), params=params))
        return WorkerOutcome(job.id, JobStatus.FAILED, str(error.code))

    jobs.update(job.id, partial(_transition, status=JobStatus.RUNNING))
    try:
        result = engine.run(EngineRequest(job_id=job.id, molecule=molecule, spec=job.spec))
    except QuantumLabError as error:
        code = str(error.code)
        params = {key: str(value) for key, value in error.params.items()}
        jobs.update(job.id, partial(_mark_failed, code=code, params=params))
        return WorkerOutcome(job.id, JobStatus.FAILED, code)

    result_uri = str(jobs.save_result(job.id, result.model_dump_json(indent=2)))
    if result.final_molecule is not None:
        jobs.save_geometry(job.id, result.final_molecule.to_xyz())
    final = JobStatus.COMPLETED_WITH_WARNINGS if result.warnings else JobStatus.COMPLETED
    jobs.update(job.id, partial(_mark_finished, result_uri=result_uri, status=final))
    return WorkerOutcome(job.id, final)
