import logging
import re

from giecar_seismic.application.job_logging import JobLogLevel
from giecar_seismic.infrastructure.logging.job_execution_logger import (
    FileJobExecutionLogger,
    job_log_path,
    read_job_log,
)


def test_job_log_path_and_directory_are_created_lazily(tmp_path):
    logs_dir = tmp_path / "logs"
    assert job_log_path(logs_dir, 42) == logs_dir / "job_42.log"
    assert not logs_dir.exists()

    FileJobExecutionLogger(logs_dir).log(
        42, JobLogLevel.INFO, "JOB_CREATED", "Job criado."
    )

    assert logs_dir.is_dir()
    assert (logs_dir / "job_42.log").is_file()


def test_append_reopen_and_distinct_jobs_do_not_duplicate_lines(tmp_path):
    first = FileJobExecutionLogger(tmp_path)
    first.log(1, JobLogLevel.INFO, "JOB_CREATED", "primeiro")
    reopened = FileJobExecutionLogger(tmp_path)
    reopened.log(1, JobLogLevel.WARNING, "CANCEL_REQUESTED", "segundo")
    reopened.log(2, JobLogLevel.ERROR, "JOB_FAILED", "outro job")

    job_one = read_job_log(tmp_path, 1)
    assert job_one.count("JOB_CREATED - primeiro") == 1
    assert job_one.count("CANCEL_REQUESTED - segundo") == 1
    assert "outro job" not in job_one
    assert "JOB_FAILED - outro job" in read_job_log(tmp_path, 2)
    assert sorted(path.name for path in tmp_path.glob("*.log")) == [
        "job_1.log",
        "job_2.log",
    ]


def test_formatter_has_timestamp_level_event_and_message(tmp_path):
    FileJobExecutionLogger(tmp_path).log(
        7, JobLogLevel.ERROR, "JOB_FAILED", "Falha curta."
    )
    line = read_job_log(tmp_path, 7).strip()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ERROR "
        r"JOB_FAILED - Falha curta\.",
        line,
    )


def test_handler_is_removed_and_closed_after_each_write(tmp_path, monkeypatch):
    import giecar_seismic.infrastructure.logging.job_execution_logger as module

    handlers = []
    real_handler = logging.FileHandler

    def recording_handler(*args, **kwargs):
        handler = real_handler(*args, **kwargs)
        handlers.append(handler)
        return handler

    monkeypatch.setattr(module.logging, "FileHandler", recording_handler)
    writer = FileJobExecutionLogger(tmp_path)
    writer.log(9, JobLogLevel.INFO, "RUN_STARTED", "Início.")
    writer.log(9, JobLogLevel.INFO, "JOB_COMPLETED", "Fim.")

    logger = logging.getLogger("giecar.job.9")
    assert logger.handlers == []
    assert len(handlers) == 2
    assert all(handler._closed and handler.stream is None for handler in handlers)


def test_missing_log_has_a_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="No execution log exists for job 99"):
        read_job_log(tmp_path, 99)
