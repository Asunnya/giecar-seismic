"""One append-only file per job, written with the standard `logging` module.

Layout: `<logs_dir>/job_<id>.log`, one line per lifecycle event:

    2026-09-12 14:02:10 INFO JOB_CREATED - Job criado com Low-pass ...

Handler strategy -- the simplest one that is leak-free and duplicate-free:
every `log()` call attaches one FileHandler (append mode) to the job's
logger `giecar.job.<id>`, writes the record, then removes and closes the
handler again. So no FileHandler outlives a call, a second
FileJobExecutionLogger over the same directory can never stack a second
handler on the same logger, and the file is released between events
(there are only a handful per job, so re-opening is negligible).
Propagation is off so lines never reach the root logger's handlers.

The attach/emit/detach section runs under one process-wide lock: the
loggers are process-global, and two threads *do* log the same job at the
same instant (the GUI thread's CANCEL_REQUESTED right after signalling
the token, the worker's JOB_CANCELLED right after observing it). Without
the lock both handlers could be attached at once and a record would be
written twice. The lock is module-level, not per instance, because every
instance shares the same `logging` registry. Subprocesses never log.

The log directory is created lazily on the first event, never at import.
"""

import logging
import threading
from pathlib import Path

from giecar_seismic.application.job_logging import JobLogLevel

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_WRITE_LOCK = threading.Lock()


def job_log_path(logs_dir: str | Path, job_id: int) -> Path:
    """The single log file of a job -- shared by every run/resume of it."""
    return Path(logs_dir) / f"job_{job_id}.log"


def read_job_log(logs_dir: str | Path, job_id: int) -> str:
    """Whole log text; FileNotFoundError with a clear message if the job
    never produced one."""
    path = job_log_path(logs_dir, job_id)
    if not path.is_file():
        raise FileNotFoundError(f"No execution log exists for job {job_id} ({path})")
    return path.read_text(encoding="utf-8")


class FileJobExecutionLogger:
    def __init__(self, logs_dir: str | Path) -> None:
        self._logs_dir = Path(logs_dir)

    def log(self, job_id: int, level: JobLogLevel, event: str, message: str) -> None:
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            logger = logging.getLogger(f"giecar.job.{job_id}")
            logger.propagate = False
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(
                job_log_path(self._logs_dir, job_id), mode="a", encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
            logger.addHandler(handler)
            try:
                logger.log(logging.getLevelName(level.value), "%s - %s", event, message)
            finally:
                logger.removeHandler(handler)
                handler.close()
