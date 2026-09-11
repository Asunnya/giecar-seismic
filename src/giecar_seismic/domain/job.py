from dataclasses import dataclass
from enum import Enum, auto


class JobStatus(Enum):
    CREATED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class InvalidTransitionError(Exception):
    pass


@dataclass
class Job:
    dataset_id: int
    cutoff_hz: float
    order: int
    status: JobStatus = JobStatus.CREATED
    error_message: str | None = None
    id: int | None = None

    def _transition(
        self, action: str, expected: JobStatus, new_status: JobStatus
    ) -> None:
        if self.status is not expected:
            raise InvalidTransitionError(
                f"cannot {action} a job in status {self.status}"
            )
        self.status = new_status

    def start(self) -> None:
        self._transition("start", JobStatus.CREATED, JobStatus.RUNNING)

    def complete(self) -> None:
        self._transition("complete", JobStatus.RUNNING, JobStatus.COMPLETED)

    def fail(self, error_message: str) -> None:
        self._transition("fail", JobStatus.RUNNING, JobStatus.FAILED)
        self.error_message = error_message

    def cancel(self) -> None:
        self._transition("cancel", JobStatus.RUNNING, JobStatus.CANCELLED)
