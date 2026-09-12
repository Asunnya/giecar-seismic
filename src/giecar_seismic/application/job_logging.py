"""Port for the per-job execution log (the assessment's optional
"log de execução persistido por job").

The application layer only knows this tiny contract: one line per
lifecycle event, addressed by job id. Where the line ends up (a file per
job under ~/.giecar-seismic/logs, in production) is infrastructure's
business. The log is observability only -- SQLite/`Job` remain the sole
source of truth for state, and the service treats every logging failure
as best-effort (see FilterJobService._log).

Deliberately no per-chunk or per-trace events: the number of lines a job
produces depends on its lifecycle, never on the survey size.
"""

from enum import Enum
from typing import Protocol


class JobLogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Event codes written into the message body ("JOB_COMPLETED - ...").
JOB_CREATED = "JOB_CREATED"
RUN_STARTED = "RUN_STARTED"
CANCEL_REQUESTED = "CANCEL_REQUESTED"
JOB_CANCELLED = "JOB_CANCELLED"
RESUME_STARTED = "RESUME_STARTED"
JOB_COMPLETED = "JOB_COMPLETED"
JOB_FAILED = "JOB_FAILED"


class JobExecutionLogger(Protocol):
    def log(self, job_id: int, level: JobLogLevel, event: str, message: str) -> None:
        """Append one event line to job `job_id`'s log. May raise; the
        caller must not let that alter the job's outcome."""
        ...


class NullJobExecutionLogger:
    """Default when no logger is injected (tests, headless composition)."""

    def log(self, job_id: int, level: JobLogLevel, event: str, message: str) -> None:
        return None
