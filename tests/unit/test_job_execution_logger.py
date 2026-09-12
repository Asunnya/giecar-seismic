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


def test_concurrent_writes_to_the_same_job_never_attach_two_handlers(
    tmp_path, monkeypatch
):
    # The GUI thread (CANCEL_REQUESTED) and the worker (JOB_CANCELLED) can
    # log the same job at the same instant. If two FileHandlers were ever
    # attached to the shared logger at once, one record would be emitted
    # through both -- a duplicated line.
    import threading

    threads, per_thread = 8, 20
    logger = logging.getLogger("giecar.job.77")
    seen_handler_counts: list[int] = []
    real_emit = logging.FileHandler.emit

    def counting_emit(self, record):
        seen_handler_counts.append(len(logger.handlers))
        real_emit(self, record)

    monkeypatch.setattr(logging.FileHandler, "emit", counting_emit)
    writer = FileJobExecutionLogger(tmp_path)
    barrier = threading.Barrier(threads)

    def work(index: int) -> None:
        barrier.wait()
        for n in range(per_thread):
            writer.log(77, JobLogLevel.INFO, "EVENT", f"t{index}-m{n}")

    workers = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()

    lines = read_job_log(tmp_path, 77).splitlines()
    assert len(lines) == threads * per_thread
    for index in range(threads):
        for n in range(per_thread):
            assert sum(line.endswith(f"EVENT - t{index}-m{n}") for line in lines) == 1
    assert set(seen_handler_counts) == {1}
    assert logger.handlers == []
