"""Жизненный цикл задания: допустимые переходы, повторы, аудит (§13, §14 ТЗ)."""

from __future__ import annotations

import pytest

from quantumlab.domain.job import Job
from quantumlab.domain.spec import CalculationSpec, PrecisionProfile, Task
from quantumlab.errors import InvalidJobTransitionError, JobNotResumableError
from quantumlab.jobs.state_machine import TERMINAL_STATUSES, JobStateMachine, JobStatus


def make_job(**overrides: object) -> Job:
    payload: dict[str, object] = {
        "name": "water-opt",
        "project_id": "proj-1",
        "owner": "tester",
        "spec": CalculationSpec(task=Task.OPTIMIZATION, profile=PrecisionProfile.STANDARD),
        "molecule_uri": "artifact://molecules/water.xyz",
        "molecule_hash": "0" * 64,
    }
    payload.update(overrides)
    return Job.model_validate(payload)


def test_happy_path_transitions() -> None:
    job = make_job()
    for status in (JobStatus.QUEUED, JobStatus.STARTING, JobStatus.RUNNING, JobStatus.COMPLETED):
        job.transition_to(status, actor="worker")
    assert job.status is JobStatus.COMPLETED
    assert job.progress.percent == 100.0
    assert job.started_at is not None
    assert job.finished_at is not None


def test_forbidden_transition_is_rejected_and_localized() -> None:
    job = make_job()
    with pytest.raises(InvalidJobTransitionError) as info:
        job.transition_to(JobStatus.RUNNING)
    assert job.status is JobStatus.DRAFT
    message = info.value.explain("ru")
    assert "Недопустимая смена состояния задания" in info.value.title("ru")
    assert "Черновик" in message or "draft" in message


def test_terminal_statuses_have_no_targets() -> None:
    for status in TERMINAL_STATUSES:
        assert JobStateMachine.available_targets(status) == ()
        assert status.is_terminal
    assert not JobStatus.RUNNING.is_terminal


def test_retry_increments_attempt() -> None:
    job = make_job()
    job.transition_to(JobStatus.QUEUED)
    job.transition_to(JobStatus.STARTING)
    job.transition_to(JobStatus.RUNNING)
    job.transition_to(JobStatus.FAILED, note="scf.not_converged")
    job.retry(actor="scheduler")
    assert job.status is JobStatus.QUEUED
    assert job.attempt == 1
    assert job.finished_at is None


def test_pause_and_resume() -> None:
    job = make_job()
    job.transition_to(JobStatus.QUEUED)
    job.transition_to(JobStatus.STARTING)
    job.transition_to(JobStatus.RUNNING)
    job.transition_to(JobStatus.PAUSED, actor="user")
    assert job.can_transition_to(JobStatus.RUNNING)
    job.transition_to(JobStatus.RUNNING, actor="user")
    assert job.status is JobStatus.RUNNING


def test_event_log_records_actor_and_order() -> None:
    job = make_job()
    job.transition_to(JobStatus.QUEUED, actor="api", note="submitted")
    job.transition_to(JobStatus.STARTING, actor="scheduler")
    assert [event.to_status for event in job.events] == [JobStatus.QUEUED, JobStatus.STARTING]
    assert job.events[0].actor == "api"
    assert job.events[0].from_status is JobStatus.DRAFT
    assert job.events[1].from_status is JobStatus.QUEUED


def test_resume_without_checkpoint_is_rejected() -> None:
    job = make_job()
    job.transition_to(JobStatus.QUEUED)
    job.transition_to(JobStatus.STARTING)
    job.transition_to(JobStatus.RUNNING)
    assert not job.is_resumable
    with pytest.raises(JobNotResumableError):
        job.ensure_resumable()


def test_resume_with_checkpoint_is_allowed() -> None:
    job = make_job()
    job.transition_to(JobStatus.QUEUED)
    job.transition_to(JobStatus.STARTING)
    job.transition_to(JobStatus.RUNNING)
    job = job.model_copy(update={"checkpoint_uri": "artifact://checkpoints/job-1-0.json"})
    assert job.is_resumable
    job.ensure_resumable()


def test_available_actions_drive_ui_buttons() -> None:
    job = make_job()
    assert JobStatus.QUEUED in job.available_actions()
    assert JobStatus.COMPLETED not in job.available_actions()
