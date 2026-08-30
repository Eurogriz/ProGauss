"""Задание (Job) — центральная сущность системы управления расчётами (§13 ТЗ).

Задание агрегирует всё, что нужно для воспроизведения и аудита: идентификатор,
владельца, проект, спецификацию, ссылку на структуру, статус с историей
переходов, ссылку на чекпоинт и на результат.

Важно: сама структура и большие артефакты в задании **не хранятся** — только
ссылки на объектное хранилище (§24 ТЗ).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from quantumlab.domain.spec import CalculationSpec, ResourceSpec
from quantumlab.errors import JobNotResumableError
from quantumlab.jobs.state_machine import JobStateMachine, JobStatus


class JobProgress(BaseModel):
    """Прогресс выполнения — источник для индикатора и ETA в очереди (§12 ТЗ)."""

    model_config = ConfigDict(extra="forbid")

    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    stage_key: str | None = Field(default=None, description="Ключ локализации этапа")
    eta_seconds: float | None = Field(default=None, ge=0.0)
    scf_iteration: int | None = Field(default=None, ge=0)
    optimization_step: int | None = Field(default=None, ge=0)


class JobEvent(BaseModel):
    """Запись журнала состояний (входит в аудит, §25 ТЗ)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    at: datetime
    from_status: JobStatus | None
    to_status: JobStatus
    actor: str = "system"
    note: str | None = None


class Job(BaseModel):
    """Расчётное задание."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    project_id: str
    owner: str
    spec: CalculationSpec
    molecule_uri: str = Field(
        description="Ссылка на структуру, например artifact://molecules/<id>.xyz"
    )
    molecule_hash: str = Field(description="SHA-256 структуры — контроль подмены на рестарте")
    status: JobStatus = JobStatus.DRAFT
    attempt: int = Field(default=0, ge=0)
    priority: int = Field(default=100, ge=0, le=1000)
    parent_job_id: str | None = Field(
        default=None, description="Исходное задание при повторном запуске"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: JobProgress = Field(default_factory=JobProgress)
    events: tuple[JobEvent, ...] = ()
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    checkpoint_uri: str | None = None
    result_uri: str | None = None
    log_uri: str | None = None
    worker_id: str | None = None
    error_code: str | None = Field(default=None, description="Код ошибки из ErrorCode")
    error_params: dict[str, str] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    # -- жизненный цикл ------------------------------------------------------ #
    def transition_to(
        self, target: JobStatus, *, actor: str = "system", note: str | None = None
    ) -> None:
        """Атомарно переводит задание в новое состояние.

        Недопустимый переход бросает :class:`InvalidJobTransitionError`, а не
        молча портит состояние. Вызывающий слой (Job Manager) обязан выполнять
        этот вызов в транзакции БД — сама модель транзакцию не открывает.
        """
        previous = self.status
        self.status = JobStateMachine.transition(previous, target)
        now = datetime.now(UTC)
        self.updated_at = now
        if target is JobStatus.STARTING and self.started_at is None:
            self.started_at = now
        if target.is_terminal or target is JobStatus.FAILED or target is JobStatus.CANCELLED:
            self.finished_at = now
        if target in (JobStatus.QUEUED, JobStatus.DRAFT):
            self.finished_at = None
        if target is JobStatus.COMPLETED or target is JobStatus.COMPLETED_WITH_WARNINGS:
            self.progress = JobProgress(percent=100.0, stage_key=self.progress.stage_key)
        self.events = (
            *self.events,
            JobEvent(at=now, from_status=previous, to_status=target, actor=actor, note=note),
        )

    def can_transition_to(self, target: JobStatus) -> bool:
        """Разрешён ли переход — для активации/скрытия кнопок в UI."""
        return JobStateMachine.can_transition(self.status, target)

    def available_actions(self) -> tuple[JobStatus, ...]:
        """Доступные сейчас переходы."""
        return JobStateMachine.available_targets(self.status)

    def retry(self, *, actor: str = "system", note: str | None = None) -> None:
        """Повторный запуск: возврат в очередь с увеличением счётчика попыток.

        Идемпотентность (§14 ТЗ): номер попытки входит в имя чекпоинта, поэтому
        повторный запуск не перезаписывает данные предыдущей попытки.
        """
        self.transition_to(JobStatus.QUEUED, actor=actor, note=note)
        self.attempt += 1

    def ensure_resumable(self) -> None:
        """Проверяет, что задание можно продолжить с чекпоинта (§14 ТЗ)."""
        if self.checkpoint_uri is None or self.status not in (
            JobStatus.PAUSED,
            JobStatus.FAILED,
            JobStatus.RUNNING,
        ):
            raise JobNotResumableError(self.id)

    # -- производные --------------------------------------------------------- #
    @property
    def elapsed_seconds(self) -> float:
        """Сколько задание уже выполняется (0, если ещё не стартовало)."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    @property
    def is_resumable(self) -> bool:
        """Есть ли контрольная точка для продолжения."""
        return self.checkpoint_uri is not None and self.status in (
            JobStatus.PAUSED,
            JobStatus.FAILED,
            JobStatus.RUNNING,
        )
