"""Persistent job history in MainWindow: loaded off the GUI thread through
JobHistoryWorker, filtered via FilterJobService.list_jobs(dataset_id,
status), newest first, with created_at always shown.
"""

import inspect
import threading
from datetime import datetime

import numpy as np
import pytest
from PyQt5.QtWidgets import QWidget

from giecar_seismic.application.filter_jobs import FilterJobService
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus
from giecar_seismic.ui import workers as workers_module
from giecar_seismic.ui.main_window import (
    CREATED_AT_COLUMN,
    DATASET_COLUMN,
    FINISHED_AT_COLUMN,
    JOBS_TABLE_HEADERS,
    PROGRESS_COLUMN,
    STATUS_COLUMN,
    MainWindow,
    format_datetime,
)
from giecar_seismic.ui.workers import JobHistoryWorker

N_SAMPLES = 64
T0 = datetime(2026, 9, 12, 10, 0, 0)  # noqa: DTZ001 -- naive, like the domain


def _at(seconds: int) -> datetime:
    return T0.replace(second=seconds)


class FakeDatasetRepository:
    def __init__(self, datasets):
        self._d = {d.id: d for d in datasets}

    def get(self, dataset_id):
        return self._d.get(dataset_id)


class RecordingJobRepository:
    """Applies list() filters itself (like the SQLite repository) and
    records every list() call with the thread it ran on."""

    def __init__(self):
        self._jobs: dict[int, Job] = {}
        self._next = 1
        self.list_calls: list[tuple[int | None, JobStatus | None, str]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail_next = False

    def add(self, job):
        job.id = self._next
        self._next += 1
        self._jobs[job.id] = job
        return job

    def get(self, job_id):
        return self._jobs.get(job_id)

    def update(self, job):
        self._jobs[job.id] = job

    def list(self, dataset_id=None, status=None):
        self.list_calls.append((dataset_id, status, threading.current_thread().name))
        self.entered.set()
        assert self.release.wait(timeout=5), "test deadlocked"
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("database is locked")
        return [
            j
            for j in self._jobs.values()
            if (dataset_id is None or j.dataset_id == dataset_id)
            and (status is None or j.status is status)
        ]


class FakeTraceReader:
    """Optionally blocks the first read until `release` is set, so a test
    can act (e.g. click Cancel) while the job is provably RUNNING."""

    def __init__(self, traces, entered=None, release=None):
        self.trace_count = len(traces)
        self._t = traces
        self._entered = entered
        self._release = release
        self._reads = 0

    def read_chunk(self, s, e):
        self._reads += 1
        if self._reads == 1 and self._entered is not None and self._release is not None:
            self._entered.set()
            assert self._release.wait(timeout=5), "test deadlocked"
        return self._t[s:e]

    def close(self):
        pass


class FakeTraceWriter:
    output_path = "/out/job.h5"

    def write_chunk(self, s, c):
        pass

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self):
        pass

    def close(self):
        pass


class RaisingWriter(FakeTraceWriter):
    def write_chunk(self, s, c):
        raise OSError("disk full")


def _dataset(dataset_id=1) -> SeismicDataset:
    return SeismicDataset(
        id=dataset_id,
        name=f"survey{dataset_id}",
        source_path=f"/surveys/survey{dataset_id}.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=4,
        n_samples=N_SAMPLES,
        sample_rate_ms=4.0,
    )


def _seed(
    repo: RecordingJobRepository,
    dataset_id,
    status,
    created,
    cutoff=30.0,
    progress=None,
    finished=None,
    output=None,
) -> Job:
    job = Job(
        dataset_id=dataset_id,
        cutoff_hz=cutoff,
        order=4,
        status=status,
        created_at=created,
        finished_at=finished,
        output_path=output,
        progress=progress
        if progress is not None
        else (100.0 if status is JobStatus.COMPLETED else 0.0),
    )
    return repo.add(job)


def _window(jobs_repo, writer=None, open_viewer=None, datasets=(1, 2), reader=None):
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    service = FilterJobService(
        datasets=FakeDatasetRepository([_dataset(i) for i in datasets]),
        jobs=jobs_repo,
        reader_factory=lambda ds: reader or FakeTraceReader(traces),
        writer_factory=lambda job, ds: writer or FakeTraceWriter(),
        chunk_size=2,
    )
    return MainWindow(service=service, open_viewer=open_viewer)


def _refresh(window, wait_for_signal):
    window.refresh_history()
    thread = window._history_thread
    assert thread is not None
    wait_for_signal(thread.finished)


def _column(window, row, column):
    return window._jobs_table.item(row, column).text()


# --- worker ------------------------------------------------------------------


def test_history_worker_never_imports_widgets():
    lines = [
        l
        for l in inspect.getsource(workers_module).splitlines()
        if l.startswith(("import ", "from "))
    ]
    assert not any("QtWidgets" in l for l in lines)


def test_history_worker_emits_the_jobs_or_the_failure():
    repo = RecordingJobRepository()
    _seed(repo, 1, JobStatus.CREATED, _at(1))
    service = FilterJobService(datasets=FakeDatasetRepository([_dataset()]), jobs=repo)
    got: list[list[Job]] = []
    worker = JobHistoryWorker(service, 1, JobStatus.CREATED)
    labels: list[dict[int, str]] = []
    worker.succeeded.connect(
        lambda jobs, names: (got.append(jobs), labels.append(names))
    )
    worker.run()
    assert [j.id for j in got[-1]] == [1]
    # dataset labels are resolved off the GUI thread too, file name only
    assert labels[-1] == {1: "survey1.segy"}
    assert repo.list_calls[-1][:2] == (1, JobStatus.CREATED)

    repo.fail_next = True
    failures: list[str] = []
    bad = JobHistoryWorker(service, None, None)
    bad.failed.connect(failures.append)
    bad.run()
    assert failures == ["database is locked"]


def test_dataset_labels_use_the_file_name_and_disambiguate_collisions():
    from giecar_seismic.ui.main_window import dataset_labels

    same_name_a = SeismicDataset(
        id=1,
        name="a",
        source_path="/x/survey.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=1,
        n_samples=1,
        sample_rate_ms=4.0,
    )
    same_name_b = SeismicDataset(
        id=2,
        name="b",
        source_path="/y/survey.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=1,
        n_samples=1,
        sample_rate_ms=4.0,
    )
    other = SeismicDataset(
        id=3,
        name="c",
        source_path="/z/other.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=1,
        n_samples=1,
        sample_rate_ms=4.0,
    )

    assert dataset_labels({1: same_name_a, 2: same_name_b, 3: other}) == {
        1: "survey.segy (#1)",
        2: "survey.segy (#2)",
        3: "other.segy",
    }
    # a job whose dataset row is gone still gets a readable label
    assert dataset_labels({}, missing=[9]) == {9: "Dataset #9"}


def test_dataset_filter_with_colliding_file_names_still_maps_to_the_right_id(
    qapp, wait_for_signal
):
    repo = RecordingJobRepository()
    _seed(repo, 1, JobStatus.COMPLETED, _at(1))
    _seed(repo, 2, JobStatus.COMPLETED, _at(2))
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    datasets = [_dataset(1), _dataset(2)]
    for d in datasets:
        d.source_path = f"/{d.id}/survey.segy"
    service = FilterJobService(
        datasets=FakeDatasetRepository(datasets),
        jobs=repo,
        reader_factory=lambda ds: FakeTraceReader(traces),
        writer_factory=lambda job, ds: FakeTraceWriter(),
        chunk_size=2,
    )
    window = MainWindow(service=service)
    _refresh(window, wait_for_signal)
    assert _column(window, 0, DATASET_COLUMN) == "survey.segy (#2)"

    window._dataset_filter.setCurrentText("survey.segy (#2)")
    wait_for_signal(window._history_thread.finished)
    assert repo.list_calls[-1][:2] == (2, None)


def test_a_job_started_in_this_session_shows_its_dataset_file_name(
    qapp, wait_for_signal
):
    repo = RecordingJobRepository()
    window = _window(repo)
    _refresh(window, wait_for_signal)
    window.set_dataset(_dataset(2))
    window._run_button.click()
    thread = window._thread
    assert thread is not None
    # labelled from the session dataset, before any history reload
    assert _column(window, 0, DATASET_COLUMN) == "survey2.segy"
    assert window._dataset_filter.findText("survey2.segy") != -1
    wait_for_signal(thread.finished)
    wait_for_signal(window.history_refreshed)  # terminal state -> reload
    assert _column(window, 0, DATASET_COLUMN) == "survey2.segy"


# --- table content -------------------------------------------------------------


def test_table_columns_include_created_at():
    assert JOBS_TABLE_HEADERS == [
        "ID",
        "Dataset",
        "Filter",
        "Cutoff(s)",
        "Order",
        "Status",
        "Progress",
        "Created at",
        "Finished at",
    ]


def test_format_datetime_is_stable_and_handles_none():
    stamp = datetime(2026, 9, 12, 10, 58, 3, 123456)  # noqa: DTZ001 -- naive, like the domain
    assert format_datetime(stamp) == "2026-09-12 10:58:03"
    assert format_datetime(None) == "-"


def test_empty_history_leaves_the_table_empty(qapp, wait_for_signal):
    window = _window(RecordingJobRepository())
    _refresh(window, wait_for_signal)
    assert window._jobs_table.rowCount() == 0
    assert window._history_status_label.text() == "0 jobs"


def test_history_is_loaded_off_the_gui_thread_with_created_at_and_newest_first(
    qapp, wait_for_signal
):
    repo = RecordingJobRepository()
    _seed(repo, 1, JobStatus.COMPLETED, _at(5), finished=_at(9), output="/out/1.h5")
    _seed(repo, 2, JobStatus.FAILED, _at(20), progress=42.5, finished=_at(21))
    _seed(repo, 1, JobStatus.CREATED, _at(20))  # same created_at as id 2 -> id desc
    window = _window(repo)

    _refresh(window, wait_for_signal)

    assert repo.list_calls[-1][2] != threading.current_thread().name
    assert [_column(window, r, 0) for r in range(3)] == ["3", "2", "1"]
    assert _column(window, 0, CREATED_AT_COLUMN) == "2026-09-12 10:00:20"
    assert _column(window, 2, CREATED_AT_COLUMN) == "2026-09-12 10:00:05"
    assert _column(window, 2, FINISHED_AT_COLUMN) == "2026-09-12 10:00:09"
    assert _column(window, 0, FINISHED_AT_COLUMN) == "-"
    assert _column(window, 1, PROGRESS_COLUMN) == "42.5%"
    assert _column(window, 2, PROGRESS_COLUMN) == "100%"
    assert _column(window, 1, STATUS_COLUMN) == "FAILED"
    assert _column(window, 1, DATASET_COLUMN) == "survey2.segy"
    assert window._history_status_label.text() == "3 jobs"


def test_startup_schedules_a_history_load_once_the_event_loop_runs(
    qapp, wait_for_signal
):
    repo = RecordingJobRepository()
    _seed(repo, 1, JobStatus.COMPLETED, _at(1), output="/out/1.h5")
    window = _window(repo)
    assert window._jobs_table.rowCount() == 0  # nothing queried synchronously

    wait_for_signal(window.history_refreshed)

    assert window._jobs_table.rowCount() == 1
    assert repo.list_calls[0][:2] == (None, None)


# --- filters -------------------------------------------------------------------


def _seeded_window(open_viewer=None):
    repo = RecordingJobRepository()
    _seed(repo, 1, JobStatus.COMPLETED, _at(1), output="/out/1.h5")
    _seed(repo, 2, JobStatus.COMPLETED, _at(2), output="/out/2.h5")
    _seed(
        repo,
        2,
        JobStatus.CANCELLED,
        _at(3),
        progress=40.0,
        finished=_at(4),
        output="/out/3.h5",
    )
    _seed(repo, 1, JobStatus.FAILED, _at(4), finished=_at(5))
    return _window(repo, open_viewer=open_viewer), repo


def test_dataset_filter_maps_to_dataset_id(qapp, wait_for_signal):
    window, repo = _seeded_window()
    _refresh(window, wait_for_signal)
    assert [
        window._dataset_filter.itemText(i)
        for i in range(window._dataset_filter.count())
    ] == [
        "All datasets",
        "survey1.segy",
        "survey2.segy",
    ]

    window._dataset_filter.setCurrentText("survey2.segy")
    wait_for_signal(window._history_thread.finished)

    assert repo.list_calls[-1][:2] == (2, None)
    assert [_column(window, r, 0) for r in range(window._jobs_table.rowCount())] == [
        "3",
        "2",
    ]


def test_status_filter_maps_to_job_status(qapp, wait_for_signal):
    window, repo = _seeded_window()
    _refresh(window, wait_for_signal)

    window._status_filter.setCurrentText("COMPLETED")
    wait_for_signal(window._history_thread.finished)

    assert repo.list_calls[-1][:2] == (None, JobStatus.COMPLETED)
    assert [_column(window, r, 0) for r in range(window._jobs_table.rowCount())] == [
        "2",
        "1",
    ]


def test_combined_filters_map_to_both_arguments(qapp, wait_for_signal):
    window, repo = _seeded_window()
    _refresh(window, wait_for_signal)
    window._dataset_filter.setCurrentText("survey1.segy")
    wait_for_signal(window._history_thread.finished)

    window._status_filter.setCurrentText("FAILED")
    wait_for_signal(window._history_thread.finished)

    assert repo.list_calls[-1][:2] == (1, JobStatus.FAILED)
    assert [_column(window, r, 0) for r in range(window._jobs_table.rowCount())] == [
        "4"
    ]
    # filtering never touches timestamps: same created_at as persisted
    assert _column(window, 0, CREATED_AT_COLUMN) == format_datetime(
        repo.get(4).created_at
    )


def test_refresh_rebuilds_the_table_without_duplicating_rows(qapp, wait_for_signal):
    window, _ = _seeded_window()
    _refresh(window, wait_for_signal)
    _refresh(window, wait_for_signal)
    window._refresh_button.click()
    wait_for_signal(window._history_thread.finished)

    assert window._jobs_table.rowCount() == 4
    assert len({_column(window, r, 0) for r in range(4)}) == 4


def test_filters_and_refresh_are_disabled_while_a_query_is_running(
    qapp, wait_for_signal
):
    window, repo = _seeded_window()
    _refresh(window, wait_for_signal)
    repo.release.clear()
    repo.entered.clear()

    window._refresh_button.click()
    assert repo.entered.wait(timeout=5)
    assert window._refresh_button.isEnabled() is False
    assert window._dataset_filter.isEnabled() is False
    assert window._status_filter.isEnabled() is False

    thread = window._history_thread
    repo.release.set()
    wait_for_signal(thread.finished)
    assert window._refresh_button.isEnabled() is True
    assert window._dataset_filter.isEnabled() is True


def test_query_failure_is_reported_and_controls_recover(qapp, wait_for_signal):
    window, repo = _seeded_window()
    _refresh(window, wait_for_signal)
    repo.fail_next = True

    _refresh(window, wait_for_signal)

    assert "database is locked" in window._history_status_label.text()
    assert window._refresh_button.isEnabled() is True
    assert window._jobs_table.rowCount() == 4  # last good content kept


# --- session jobs reconcile with persisted history ---------------------------


@pytest.mark.parametrize(
    "writer,cancel,expected",
    [
        (None, False, JobStatus.COMPLETED),
        (RaisingWriter(), False, JobStatus.FAILED),
        (None, True, JobStatus.CANCELLED),
    ],
    ids=["completed", "failed", "cancelled"],
)
def test_session_job_appears_and_terminal_state_is_reconciled_from_history(
    qapp, wait_for_signal, writer, cancel, expected
):
    repo = RecordingJobRepository()
    _seed(repo, 1, JobStatus.COMPLETED, _at(1), output="/out/1.h5")
    entered, release = threading.Event(), threading.Event()
    reader = FakeTraceReader(
        np.zeros((4, N_SAMPLES), dtype=np.float32), entered, release
    )
    window = _window(repo, writer=writer, reader=reader)
    _refresh(window, wait_for_signal)
    window.set_dataset(_dataset(1))

    window._run_button.click()
    assert (
        window._jobs_table.rowCount() == 2 and _column(window, 0, 0) == "2"
    )  # newest on top
    assert entered.wait(timeout=5)  # job is RUNNING, blocked in its first read
    if cancel:
        window._cancel_button.click()
    release.set()
    wait_for_signal(window._thread.finished)
    wait_for_signal(window.history_refreshed)  # terminal state triggers a refresh

    assert window._jobs_table.rowCount() == 2
    assert _column(window, 0, STATUS_COLUMN) == expected.name
    assert _column(window, 0, CREATED_AT_COLUMN) == format_datetime(
        repo.get(2).created_at
    )
    assert _column(window, 0, FINISHED_AT_COLUMN) == format_datetime(
        repo.get(2).finished_at
    )
    assert _column(window, 0, PROGRESS_COLUMN) == f"{repo.get(2).progress:g}%"


# --- view output on historical jobs --------------------------------------------


def test_historical_completed_job_enables_view_output_without_the_session_dataset(
    qapp, wait_for_signal
):
    opened: list[int] = []

    def open_viewer(job_id: int, parent: QWidget) -> QWidget:
        opened.append(job_id)
        return QWidget(parent)

    window, _ = _seeded_window(open_viewer=open_viewer)
    _refresh(window, wait_for_signal)
    assert window._dataset is None

    window._jobs_table.selectRow(2)  # id 2: COMPLETED, dataset 2
    assert window._view_output_button.isEnabled() is True
    assert window._open_output_button.isEnabled() is True
    window._view_output_button.click()
    assert opened == [2]

    window._jobs_table.selectRow(1)  # id 3: CANCELLED (has output) -> not viewable
    assert window._view_output_button.isEnabled() is False
    window._jobs_table.selectRow(0)  # id 4: FAILED, no output
    assert window._view_output_button.isEnabled() is False
    assert window._open_output_button.isEnabled() is False
