from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class FilterType(Enum):
    LOW_PASS = auto()
    HIGH_PASS = auto()
    BAND_PASS = auto()


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
    # 0..100, monotonic while RUNNING. The Job only *records* progress;
    # how it is computed (processed / physical trace_count) is the
    # service's business, not the domain's.
    progress: float = 0.0
    # None until a writer actually produces an output for this job.
    output_path: str | None = None
    # Lifecycle timestamps: created_at is set at construction, started_at
    # by start(), finished_at by whichever terminal transition happens.
    # One finished_at + status is enough to know how and when a job
    # ended -- per-status terminal timestamps would just be redundant.
    # Naive local datetimes, consistent with SeismicDataset.created_at.
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    id: int | None = None
    filter_type: FilterType = FilterType.LOW_PASS
    upper_cutoff_hz: float | None = None

    processed_traces: int = 0
    resume_count: int = 0

    def __post_init__(self) -> None:
        for name in ("processed_traces", "resume_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def resume(self) -> None:
        """Creativity track's sole additional transition; preserve logical start."""
        self._transition("resume", JobStatus.CANCELLED, JobStatus.RUNNING)
        self.finished_at = None
        self.resume_count += 1

    def record_checkpoint(self, processed_traces: int, total_traces: int) -> None:
        """Record a confirmed durable prefix, never a rounded percentage."""
        if self.status is not JobStatus.RUNNING:
            raise InvalidTransitionError("checkpoint requires a RUNNING job")
        if (
            type(processed_traces) is not int
            or total_traces <= 0
            or not self.processed_traces <= processed_traces <= total_traces
        ):
            raise ValueError(
                "processed_traces must be monotonic and within trace count"
            )
        self.advance_progress(100 * processed_traces / total_traces)
        self.processed_traces = processed_traces

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
        self.started_at = datetime.now()  # noqa: DTZ005 -- naive local time, same policy as SeismicDataset.created_at

    def complete(self) -> None:
        self._transition("complete", JobStatus.RUNNING, JobStatus.COMPLETED)
        self.progress = 100.0
        self.finished_at = datetime.now()  # noqa: DTZ005 -- naive local time, same policy as SeismicDataset.created_at

    def fail(self, error_message: str) -> None:
        self._transition("fail", JobStatus.RUNNING, JobStatus.FAILED)
        self.error_message = error_message
        self.finished_at = datetime.now()  # noqa: DTZ005 -- naive local time, same policy as SeismicDataset.created_at

    def cancel(self) -> None:
        self._transition("cancel", JobStatus.RUNNING, JobStatus.CANCELLED)
        self.finished_at = datetime.now()  # noqa: DTZ005 -- naive local time, same policy as SeismicDataset.created_at

    def advance_progress(self, progress: float) -> None:
        """Record progress reached so far (0..100) while RUNNING.

        Never regresses: a value below the current one is rejected, so a
        late or out-of-order update can't make the job look like it went
        backwards. Terminal states keep whatever was last reached (or 100
        for COMPLETED, set by complete()).
        """
        if self.status is not JobStatus.RUNNING:
            raise InvalidTransitionError(
                f"cannot advance progress of a job in status {self.status}"
            )
        if not 0 <= progress <= 100:
            raise ValueError(f"progress must be within 0..100, got {progress}")
        if progress < self.progress:
            raise ValueError(
                f"progress cannot regress from {self.progress} to {progress}"
            )
        self.progress = float(progress)
