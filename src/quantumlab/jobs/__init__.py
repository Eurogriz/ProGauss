"""Подсистема управления заданиями (Job Manager, §13 ТЗ)."""

from quantumlab.jobs.state_machine import ALLOWED_TRANSITIONS, JobStateMachine, JobStatus

__all__ = ["ALLOWED_TRANSITIONS", "JobStateMachine", "JobStatus"]
