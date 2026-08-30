"""Машина состояний задания (§13, §14 ТЗ).

Требование production-надёжности: смена состояния — это **атомарная операция с
валидацией**, а не присваивание поля. Недопустимый переход (например,
«Завершён» → «Выполняется») отклоняется явно, а не приводит систему в
несогласованное состояние после падения worker'а.

Диаграмма переходов::

    DRAFT ──► QUEUED ──► STARTING ──► RUNNING ──► COMPLETED
      │           │           │          │  ├───► COMPLETED_WITH_WARNINGS
      │           │           │          │  ├───► PAUSED ──► RUNNING
      │           │           │          │  ├───► FAILED ──► QUEUED  (retry)
      ▼           ▼           ▼          ▼
    CANCELLED ◄───┴───────────┴──────────┘   CANCELLED ──► QUEUED (re-queue)

Терминальные состояния — COMPLETED и COMPLETED_WITH_WARNINGS: из них задание
не возобновляется, повторный запуск создаёт новое задание со ссылкой
``parent_job_id``.
FAILED и CANCELLED — полу-терминальные: их можно вернуть в очередь.
"""

from __future__ import annotations

from enum import StrEnum

from quantumlab.errors import InvalidJobTransitionError


class JobStatus(StrEnum):
    """Состояния задания (§13 ТЗ). Значения — часть контракта API и БД."""

    DRAFT = "draft"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

    @property
    def i18n_key(self) -> str:
        """Ключ локализации статуса (например ``status.running``)."""
        return f"status.{self.value}"

    @property
    def is_terminal(self) -> bool:
        """Задание завершено и больше не исполняется."""
        return self in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        """Задание занимает вычислительные ресурсы."""
        return self in (JobStatus.STARTING, JobStatus.RUNNING)


#: Состояния, из которых невозможен никакой переход.
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}
)

#: Допустимые переходы. Ключ — текущее состояние, значение — множество целевых.
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.DRAFT: frozenset({JobStatus.QUEUED, JobStatus.CANCELLED}),
    JobStatus.QUEUED: frozenset({JobStatus.STARTING, JobStatus.CANCELLED}),
    JobStatus.STARTING: frozenset({JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.PAUSED,
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_WITH_WARNINGS,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.PAUSED: frozenset({JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.FAILED: frozenset({JobStatus.QUEUED}),
    JobStatus.CANCELLED: frozenset({JobStatus.QUEUED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.COMPLETED_WITH_WARNINGS: frozenset(),
}


class JobStateMachine:
    """Проверка и выполнение переходов состояния задания.

    Класс не хранит состояния: он чистый «валидатор правил», поэтому его можно
    использовать и в worker'е, и в API-сервере, и в CLI — правила не разъедутся.
    """

    @staticmethod
    def can_transition(current: JobStatus, target: JobStatus) -> bool:
        """Разрешён ли переход ``current`` → ``target``."""
        return target in ALLOWED_TRANSITIONS[current]

    @staticmethod
    def transition(current: JobStatus, target: JobStatus) -> JobStatus:
        """Выполняет переход или бросает :class:`InvalidJobTransitionError`."""
        if not JobStateMachine.can_transition(current, target):
            raise InvalidJobTransitionError(current.value, target.value)
        return target

    @staticmethod
    def available_targets(current: JobStatus) -> tuple[JobStatus, ...]:
        """Куда задание можно перевести сейчас — источник для кнопок в UI."""
        return tuple(sorted(ALLOWED_TRANSITIONS[current], key=lambda status: status.value))

    @staticmethod
    def is_retryable(current: JobStatus) -> bool:
        """Можно ли вернуть задание в очередь (retry policy, §14 ТЗ)."""
        return JobStatus.QUEUED in ALLOWED_TRANSITIONS[current]
