"""'View log' in the history: read off the GUI thread by JobLogWorker, shown in
a read-only dialog; a missing or unreadable log is reported, never fatal."""

import inspect
import threading

import numpy as np
from PyQt5.QtWidgets import QPlainTextEdit

from giecar_seismic.application.filter_jobs import FilterJobService
from giecar_seismic.domain.job import JobStatus
from giecar_seismic.ui import workers as workers_module
from giecar_seismic.ui.main_window import JobLogDialog, MainWindow
from giecar_seismic.ui.workers import JobLogWorker
from tests.unit.test_job_history import (
    FakeDatasetRepository,
    FakeTraceReader,
    FakeTraceWriter,
    RecordingJobRepository,
    _at,
    _dataset,
    _refresh,
    _seed,
)

N_SAMPLES = 64


class RecordingLogReader:
    def __init__(self, content=None, error=None):
        self.content, self.error = content, error
        self.calls: list[tuple[int, str]] = []

    def __call__(self, job_id: int) -> str:
        self.calls.append((job_id, threading.current_thread().name))
        if self.error is not None:
            raise self.error
        return self.content


def _window(reader, wait_for_signal):
    repo = RecordingJobRepository()
    _seed(repo, 1, JobStatus.COMPLETED, _at(1), output="/out/1.h5")
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    service = FilterJobService(
        datasets=FakeDatasetRepository([_dataset(1)]),
        jobs=repo,
        reader_factory=lambda ds: FakeTraceReader(traces),
        writer_factory=lambda job, ds: FakeTraceWriter(),
        chunk_size=2,
    )
    window = MainWindow(service=service, read_job_log=reader)
    _refresh(window, wait_for_signal)
    return window


def _click_view_log(window, wait_for_signal):
    window._view_log_button.click()
    thread = window._log_thread
    assert thread is not None
    wait_for_signal(thread.finished)


def test_log_worker_never_imports_widgets_and_emits_content_or_failure():
    lines = inspect.getsource(workers_module).splitlines()
    assert not any(line.startswith("from PyQt5.QtWidgets") for line in lines)

    got, failures = [], []
    worker = JobLogWorker(RecordingLogReader("2026 INFO JOB_CREATED - ok\n"), 7)
    worker.loaded.connect(got.append)
    worker.run()
    assert got == ["2026 INFO JOB_CREATED - ok\n"]

    bad = JobLogWorker(RecordingLogReader(error=FileNotFoundError("no log for 7")), 7)
    bad.failed.connect(failures.append)
    bad.run()
    assert failures == ["no log for 7"]


def test_view_log_is_disabled_without_selection_and_enabled_with_one(
    qapp, wait_for_signal
):
    window = _window(RecordingLogReader("x"), wait_for_signal)
    assert window._view_log_button.isEnabled() is False
    window._jobs_table.selectRow(0)
    assert window._view_log_button.isEnabled() is True
    window._jobs_table.clearSelection()
    assert window._view_log_button.isEnabled() is False


def test_view_log_without_a_reader_stays_disabled(qapp, wait_for_signal):
    window = _window(None, wait_for_signal)
    window._jobs_table.selectRow(0)
    assert window._view_log_button.isEnabled() is False


def test_view_log_reads_off_the_gui_thread_and_fills_a_read_only_dialog(
    qapp, wait_for_signal
):
    reader = RecordingLogReader("2026-09-12 14:02:10 INFO JOB_CREATED - Job criado.\n")
    window = _window(reader, wait_for_signal)
    window._jobs_table.selectRow(0)

    _click_view_log(window, wait_for_signal)

    assert reader.calls[0][0] == 1
    assert reader.calls[0][1] != threading.current_thread().name
    dialog = window._log_dialog
    assert isinstance(dialog, JobLogDialog)
    text = dialog.findChild(QPlainTextEdit)
    assert text.isReadOnly()
    assert "JOB_CREATED - Job criado." in text.toPlainText()
    assert "1" in dialog.windowTitle()
    assert window._view_log_button.isEnabled() is True  # controls recovered


def test_missing_log_shows_a_clear_message_without_blocking(qapp, wait_for_signal):
    reader = RecordingLogReader(
        error=FileNotFoundError("No execution log exists for job 1")
    )
    window = _window(reader, wait_for_signal)
    window._jobs_table.selectRow(0)

    _click_view_log(window, wait_for_signal)

    text = window._log_dialog.findChild(QPlainTextEdit).toPlainText()
    assert "No execution log exists for job 1" in text
    assert window._log_thread is None
    assert window._view_log_button.isEnabled() is True
    assert window.isEnabled()


def test_read_error_does_not_block_the_window(qapp, wait_for_signal):
    reader = RecordingLogReader(error=PermissionError("logs are unreadable"))
    window = _window(reader, wait_for_signal)
    window._jobs_table.selectRow(0)

    _click_view_log(window, wait_for_signal)

    assert (
        "logs are unreadable"
        in window._log_dialog.findChild(QPlainTextEdit).toPlainText()
    )
    assert window._view_log_button.isEnabled() is True
