import inspect
import threading

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox

from giecar_seismic.application.filter_jobs import (
    FilterJobService,
    InvalidFilterParametersError,
)
from giecar_seismic.application.seismic_viewer import FilterPreview
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import FilterType, Job, JobStatus
from giecar_seismic.ui import main_window as main_window_module
from giecar_seismic.ui.main_window import (
    JOBS_TABLE_HEADERS,
    STATUS_COLUMN,
    JobLogDialog,
    MainWindow,
)

N_SAMPLES = 64  # large enough for sosfiltfilt's padlen at order=4


class FakeDatasetRepository:
    def __init__(self, datasets: list[SeismicDataset]):
        self._datasets = {dataset.id: dataset for dataset in datasets}

    def get(self, dataset_id: int) -> SeismicDataset | None:
        return self._datasets.get(dataset_id)


class FakeJobRepository:
    def __init__(self) -> None:
        self._jobs: dict[int, Job] = {}
        self._next_id = 1

    def add(self, job: Job) -> Job:
        job.id = self._next_id
        self._jobs[job.id] = job
        self._next_id += 1
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job: Job) -> None:
        assert job.id is not None
        self._jobs[job.id] = job

    def list(self, dataset_id=None, status=None):
        return list(self._jobs.values())


class FakeTraceReader:
    def __init__(self, traces: np.ndarray):
        self.trace_count = len(traces)
        self._traces = traces

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        return self._traces[start:stop]

    def close(self) -> None:
        pass


class FakeTraceWriter:
    output_path = "/fake/output.h5"

    def __init__(self) -> None:
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self.finalized = True

    def close(self) -> None:
        pass


class RaisingWriteChunkTraceWriter:
    output_path = "/fake/output.h5"

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        raise OSError("disk full")

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        pass

    def close(self) -> None:
        pass


class BlockingFirstReadTraceReader:
    """Blocks read_chunk() the first time it's called, until `release` is
    set -- used to pin a real QThread's worker mid-run so a test can
    click Cancel while it's guaranteed to still be RUNNING, without any
    sleep()s.
    """

    def __init__(
        self, traces: np.ndarray, entered: threading.Event, release: threading.Event
    ):
        self.trace_count = len(traces)
        self._traces = traces
        self._entered = entered
        self._release = release
        self._read_count = 0

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        self._read_count += 1
        if self._read_count == 1:
            self._entered.set()
            assert self._release.wait(timeout=5), "test deadlocked: release never set"
        return self._traces[start:stop]

    def close(self) -> None:
        pass


def _dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=4,
        n_samples=N_SAMPLES,
        sample_rate_ms=4.0,
    )


def _wait_settled(window, wait_for_signal, signal) -> None:
    """Wait for `signal`, then also drain any history query MainWindow
    started meanwhile (startup load or the post-terminal reconcile), so no
    QThread outlives the test."""
    wait_for_signal(signal)
    while window._history_thread is not None:
        wait_for_signal(window._history_thread.finished)


def _build_service(dataset: SeismicDataset, reader, writer, chunk_size: int = 1):
    return FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda ds: reader,
        writer_factory=lambda job, dataset: writer,
        chunk_size=chunk_size,
    )


def test_construction_builds_the_expected_sections(qapp):
    window = MainWindow()

    group_titles = {group.title() for group in window.findChildren(QGroupBox)}
    assert group_titles == {"Dataset", "Filter", "Processing", "Jobs"}

    assert window._segy_path_edit.text() == ""
    assert window._select_segy_button.text() == "Select SEG-Y"
    assert window._cutoff_spinbox.value() == 30.0
    assert window._order_spinbox.value() == 4
    assert window._run_button.text() == "Run Filter"
    assert window._cancel_button.text() == "Cancel"
    assert window._status_label.text() == "Idle"
    assert window._jobs_table.columnCount() == len(JOBS_TABLE_HEADERS)
    assert window._open_output_button.text() == "Open output folder"
    assert window._open_output_button.isEnabled() is False


def test_initial_state_disables_run_and_cancel_with_zero_progress(qapp):
    window = MainWindow(service=None)

    assert window._run_button.isEnabled() is False
    assert window._cancel_button.isEnabled() is False
    assert window._progress_bar.value() == 0


def test_run_becomes_enabled_once_service_and_dataset_are_both_set(qapp):
    dataset = _dataset()
    service = _build_service(
        dataset, FakeTraceReader(np.zeros((1, N_SAMPLES))), FakeTraceWriter()
    )
    window = MainWindow(service=service)
    assert window._run_button.isEnabled() is False  # service set, but no dataset yet

    window.set_dataset(dataset)

    assert window._run_button.isEnabled() is True
    assert window._name_label.text() == dataset.name
    assert window._inlines_label.text() == str(dataset.n_inlines)
    assert window._traces_label.text() == str(dataset.n_traces)
    assert window._samples_label.text() == str(dataset.n_samples)


def test_on_progress_updates_the_progress_bar(qapp):
    window = MainWindow()

    window._on_progress(42)

    assert window._progress_bar.value() == 42


def test_starting_a_job_disables_run_and_enables_cancel_immediately(
    qapp, wait_for_signal
):
    dataset = _dataset()
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    service = _build_service(
        dataset, FakeTraceReader(traces), FakeTraceWriter(), chunk_size=2
    )
    window = MainWindow(service=service)
    window.set_dataset(dataset)

    window._run_button.click()

    # No Qt event loop has run yet, so no cross-thread (queued) signal
    # could possibly have been delivered -- this is deterministic
    # regardless of how fast the worker thread finishes in the background.
    assert window._run_button.isEnabled() is False
    assert window._cancel_button.isEnabled() is True
    thread_ref = window._thread
    assert thread_ref is not None

    # let the thread actually finish before the test ends: a QThread
    # destroyed while still running crashes the process. Waiting on
    # thread.finished (not just the worker's terminal signal) is what
    # actually guarantees the OS thread is done.
    _wait_settled(window, wait_for_signal, thread_ref.finished)


def test_completed_job_restores_controls_and_updates_the_jobs_table(
    qapp, wait_for_signal
):
    dataset = _dataset()
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    writer = FakeTraceWriter()
    service = _build_service(dataset, FakeTraceReader(traces), writer, chunk_size=2)
    window = MainWindow(service=service)
    window.set_dataset(dataset)

    window._run_button.click()
    thread_ref = window._thread
    assert thread_ref is not None

    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._run_button.isEnabled() is True
    assert window._cancel_button.isEnabled() is False
    assert window._progress_bar.value() == 100
    assert window._status_label.text() == "Completed: /fake/output.h5"
    assert writer.finalized is True

    assert window._jobs_table.rowCount() == 1
    assert window._jobs_table.item(0, STATUS_COLUMN).text() == JobStatus.COMPLETED.name

    # cleanup: the QThread/worker references must be dropped, but only
    # after the thread actually finished -- never while it was running.
    assert window._thread is None
    assert window._worker is None


def test_failed_job_reaches_the_ui_through_a_signal_and_is_not_swallowed(
    qapp, wait_for_signal
):
    dataset = _dataset()
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    service = _build_service(
        dataset, FakeTraceReader(traces), RaisingWriteChunkTraceWriter(), chunk_size=2
    )
    window = MainWindow(service=service)
    window.set_dataset(dataset)

    window._run_button.click()
    thread_ref = window._thread
    assert thread_ref is not None

    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._status_label.text() == "Failed: disk full"
    assert window._run_button.isEnabled() is True
    assert window._cancel_button.isEnabled() is False
    assert window._jobs_table.item(0, STATUS_COLUMN).text() == JobStatus.FAILED.name


def test_cancel_requests_cooperative_cancellation_without_touching_the_thread(
    qapp, wait_for_signal
):
    # The main_window module must never kill/terminate a QThread directly
    # -- cancellation goes exclusively through the application's
    # cooperative mechanism (FilterJobService.cancel_job()).
    source = inspect.getsource(main_window_module)
    assert "terminate(" not in source
    assert ".kill(" not in source

    dataset = _dataset()
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    entered = threading.Event()
    release = threading.Event()
    reader = BlockingFirstReadTraceReader(traces, entered, release)
    writer = FakeTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=1)
    window = MainWindow(service=service)
    window.set_dataset(dataset)

    window._run_button.click()
    thread_ref = window._thread
    assert thread_ref is not None

    # block the test thread (plain threading.Event, no Qt event loop
    # needed) until the worker thread is genuinely mid-run.
    assert entered.wait(timeout=5), "test deadlocked: worker never started reading"

    window._cancel_button.click()

    # cooperative cancellation must not touch the thread itself: it is
    # still the very same, still-running QThread right after Cancel.
    assert window._thread is thread_ref
    assert thread_ref.isRunning() is True

    release.set()
    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._thread is None
    assert window._worker is None
    [job] = service.list_jobs()
    assert job.status is JobStatus.CANCELLED
    assert writer.finalized is False
    assert window._jobs_table.item(0, STATUS_COLUMN).text() == JobStatus.CANCELLED.name


# --- Select SEG-Y -> import worker -> Dataset section --------------------


def test_selecting_a_file_starts_import_without_running_it_on_the_gui_thread(
    qapp, wait_for_signal, monkeypatch
):
    dataset = _dataset()
    entered = threading.Event()
    release = threading.Event()

    def blocking_importer(source_path: str, name: str) -> SeismicDataset:
        entered.set()
        assert release.wait(timeout=5), "test deadlocked: release never set"
        return dataset

    window = MainWindow(dataset_importer=blocking_importer)
    monkeypatch.setattr(window, "_prompt_for_segy_path", lambda: "/data/survey.segy")

    window._select_segy_button.click()

    # the click handler returns immediately -- a QThread was started, and
    # nothing here waited for the (currently blocked) import to finish.
    # That is only possible if the importer is not running on this (the
    # GUI/test) thread.
    assert window._segy_path_edit.text() == "/data/survey.segy"
    thread_ref = window._import_thread
    assert thread_ref is not None
    assert window._select_segy_button.isEnabled() is False
    assert window._status_label.text() == "Loading dataset..."
    assert window._dataset is None  # not yet imported

    assert entered.wait(timeout=5), "test deadlocked: importer never started running"

    release.set()
    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._dataset is dataset
    assert window._name_label.text() == dataset.name
    assert window._inlines_label.text() == str(dataset.n_inlines)
    assert window._crosslines_label.text() == str(dataset.n_crosslines)
    assert window._samples_label.text() == str(dataset.n_samples)
    assert window._status_label.text() == "Dataset loaded"
    assert window._import_thread is None
    assert window._import_worker is None
    assert window._select_segy_button.isEnabled() is True


def test_cancelling_the_file_dialog_does_nothing(qapp, monkeypatch):
    window = MainWindow()
    monkeypatch.setattr(window, "_prompt_for_segy_path", lambda: "")

    window._select_segy_button.click()

    assert window._segy_path_edit.text() == ""
    assert window._import_thread is None
    assert window._status_label.text() == "Idle"


def test_dataset_import_failure_reaches_the_ui_without_a_partial_dataset(
    qapp, wait_for_signal, monkeypatch
):
    def raising_importer(source_path: str, name: str) -> SeismicDataset:
        raise OSError("not a valid SEG-Y file")

    window = MainWindow(dataset_importer=raising_importer)
    monkeypatch.setattr(window, "_prompt_for_segy_path", lambda: "/data/broken.segy")

    window._select_segy_button.click()
    thread_ref = window._import_thread
    assert thread_ref is not None

    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._dataset is None
    assert (
        window._status_label.text() == "Failed to load dataset: not a valid SEG-Y file"
    )
    assert window._select_segy_button.isEnabled() is True
    assert window._import_thread is None
    assert window._import_worker is None


def test_run_button_stays_disabled_while_an_import_is_in_progress(
    qapp, wait_for_signal
):
    # A dataset from a *previous* successful import is already set, so
    # Run would normally be enabled -- but a new import overwriting it
    # must keep Run disabled until that import settles.
    dataset = _dataset()
    entered = threading.Event()
    release = threading.Event()

    def blocking_importer(source_path: str, name: str) -> SeismicDataset:
        entered.set()
        assert release.wait(timeout=5), "test deadlocked: release never set"
        return dataset

    service = _build_service(
        dataset, FakeTraceReader(np.zeros((1, N_SAMPLES))), FakeTraceWriter()
    )
    window = MainWindow(service=service, dataset_importer=blocking_importer)
    window.set_dataset(dataset)
    assert window._run_button.isEnabled() is True

    window._start_dataset_import("/data/survey.segy")

    assert window._run_button.isEnabled() is False
    assert entered.wait(timeout=5), "test deadlocked: importer never started running"

    release.set()
    thread_ref = window._import_thread
    assert thread_ref is not None
    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._run_button.isEnabled() is True


def test_selecting_a_file_again_while_importing_is_ignored(qapp, wait_for_signal):
    dataset = _dataset()
    entered = threading.Event()
    release = threading.Event()
    import_calls: list[str] = []

    def blocking_importer(source_path: str, name: str) -> SeismicDataset:
        import_calls.append(source_path)
        entered.set()
        assert release.wait(timeout=5), "test deadlocked: release never set"
        return dataset

    window = MainWindow(dataset_importer=blocking_importer)
    window._start_dataset_import("/data/first.segy")
    first_thread = window._import_thread
    assert first_thread is not None
    assert entered.wait(timeout=5), "test deadlocked: importer never started running"

    # a second attempt while the first is still running must be ignored:
    # it must not create a second thread/worker or call the importer again.
    window._on_select_segy_clicked()
    assert window._import_thread is first_thread
    assert import_calls == ["/data/first.segy"]

    release.set()
    _wait_settled(window, wait_for_signal, first_thread.finished)


def test_close_event_is_rejected_while_import_thread_is_running_and_accepted_when_idle(
    qapp, wait_for_signal
):
    from PyQt5.QtGui import QCloseEvent

    dataset = _dataset()
    entered = threading.Event()
    release = threading.Event()

    def blocking_importer(source_path: str, name: str) -> SeismicDataset:
        entered.set()
        assert release.wait(timeout=5), "test deadlocked: release never set"
        return dataset

    window = MainWindow(dataset_importer=blocking_importer)
    window._start_dataset_import("/data/survey.segy")
    thread_ref = window._import_thread
    assert thread_ref is not None
    assert entered.wait(timeout=5), "test deadlocked: importer never started running"

    event = QCloseEvent()
    window.closeEvent(event)

    # a running QThread must never be destroyed: the close is rejected
    # rather than letting Qt tear the window (and its QThread) down.
    assert event.isAccepted() is False
    assert window._import_thread is thread_ref
    assert thread_ref.isRunning() is True

    release.set()
    _wait_settled(window, wait_for_signal, thread_ref.finished)

    idle_event = QCloseEvent()
    window.closeEvent(idle_event)
    assert idle_event.isAccepted() is True


# --- Stabilization fixes ---------------------------------------------------


def test_failed_reimport_invalidates_the_previously_loaded_dataset(
    qapp, wait_for_signal
):
    dataset_a = _dataset()
    service = _build_service(
        dataset_a, FakeTraceReader(np.zeros((1, N_SAMPLES))), FakeTraceWriter()
    )

    def failing_importer(source_path: str, name: str) -> SeismicDataset:
        raise OSError("bad file B")

    window = MainWindow(service=service, dataset_importer=failing_importer)
    window.set_dataset(dataset_a)
    assert window._run_button.isEnabled() is True

    window._start_dataset_import("/data/B.segy")

    # a new path must never end up paired with the previous dataset: it
    # is invalidated the instant the new import starts, before we even
    # know whether it will succeed.
    assert window._dataset is None
    assert window._run_button.isEnabled() is False
    assert window._name_label.text() == "-"
    assert window._traces_label.text() == "-"

    thread_ref = window._import_thread
    assert thread_ref is not None
    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._dataset is None
    assert window._run_button.isEnabled() is False
    assert window._status_label.text() == "Failed to load dataset: bad file B"
    assert window._name_label.text() == "-"


def test_dataset_without_an_id_never_enables_run(qapp):
    unpersisted_dataset = SeismicDataset(
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=4,
        n_samples=N_SAMPLES,
        sample_rate_ms=4.0,
    )
    assert unpersisted_dataset.id is None
    service = _build_service(
        unpersisted_dataset,
        FakeTraceReader(np.zeros((1, N_SAMPLES))),
        FakeTraceWriter(),
    )
    window = MainWindow(service=service)

    window.set_dataset(unpersisted_dataset)

    assert window._run_button.isEnabled() is False


def test_set_dataset_caps_the_cutoff_spinbox_below_nyquist(qapp):
    window = MainWindow()
    window._cutoff_spinbox.setValue(30.0)
    dataset = _dataset()  # sample_rate_ms=4.0 -> nyquist_hz == 125.0

    window.set_dataset(dataset)

    assert window._cutoff_spinbox.maximum() == pytest.approx(124.99)
    assert window._cutoff_spinbox.value() == pytest.approx(30.0)


def test_set_dataset_clamps_a_cutoff_value_above_the_new_nyquist_limit(qapp):
    window = MainWindow()
    window._cutoff_spinbox.setValue(100.0)
    low_nyquist_dataset = SeismicDataset(
        id=1,
        name="low",
        source_path="/data/low.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=4,
        n_samples=N_SAMPLES,
        sample_rate_ms=200.0,  # nyquist_hz == 1000/200/2 == 2.5
    )

    window.set_dataset(low_nyquist_dataset)

    assert window._cutoff_spinbox.maximum() == pytest.approx(2.49)
    assert window._cutoff_spinbox.value() == pytest.approx(2.49)


def test_select_segy_disabled_and_ignored_while_a_filter_job_is_running(
    qapp, wait_for_signal
):
    dataset = _dataset()
    entered = threading.Event()
    release = threading.Event()
    reader = BlockingFirstReadTraceReader(
        np.zeros((4, N_SAMPLES), dtype=np.float32), entered, release
    )
    writer = FakeTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=1)
    window = MainWindow(service=service, dataset_importer=lambda path, name: dataset)
    window.set_dataset(dataset)

    window._run_button.click()
    assert entered.wait(timeout=5), "test deadlocked: worker never started reading"

    assert window._select_segy_button.isEnabled() is False

    # a click while the job is running must be a no-op: it must not start
    # a competing import that would replace the dataset the running job
    # is using -- Cancel remains the only way to affect this run.
    window._on_select_segy_clicked()
    assert window._import_thread is None
    assert window._dataset is dataset

    release.set()
    thread_ref = window._thread
    assert thread_ref is not None
    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._select_segy_button.isEnabled() is True


# --- Import delivers a persisted dataset -------------------------------


def test_main_window_imports_no_sqlalchemy_or_infrastructure():
    # Check actual import statements, not prose: the UI must reach
    # persistence and SEG-Y reading only through the injected callable.
    import_lines = [
        line.strip()
        for line in inspect.getsource(main_window_module).splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert not any("sqlalchemy" in line for line in import_lines)
    assert not any("giecar_seismic.infrastructure" in line for line in import_lines)


def test_select_segy_is_disabled_when_no_importer_is_composed(qapp):
    window = MainWindow(dataset_importer=None)

    assert window._select_segy_button.isEnabled() is False


def test_successful_import_delivers_a_persisted_dataset_with_id_to_the_ui(
    qapp, wait_for_signal, monkeypatch
):
    from dataclasses import replace

    persisted = replace(_dataset(), id=42)
    importer_thread_names: list[str] = []

    def full_importer(source_path: str, name: str) -> SeismicDataset:
        # what production wires here is ImportDatasetUseCase (read + persist);
        # the window only ever sees the callable and the persisted result.
        importer_thread_names.append(threading.current_thread().name)
        return persisted

    window = MainWindow(dataset_importer=full_importer)
    monkeypatch.setattr(window, "_prompt_for_segy_path", lambda: "/data/survey.segy")

    window._select_segy_button.click()
    thread_ref = window._import_thread
    assert thread_ref is not None
    _wait_settled(window, wait_for_signal, thread_ref.finished)

    assert window._dataset is persisted
    assert window._dataset.id == 42
    assert window._traces_label.text() == str(persisted.n_traces)
    # the importer ran off the GUI thread (this test's thread)
    assert importer_thread_names != [threading.current_thread().name]


# --- Output location: visible in the table/status and openable ------------


def _completed_window(qapp, wait_for_signal, output_path="/out/job-1.h5"):
    dataset = _dataset()
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    writer = FakeTraceWriter()
    writer.output_path = output_path
    service = _build_service(dataset, FakeTraceReader(traces), writer, chunk_size=2)
    window = MainWindow(service=service)
    window.set_dataset(dataset)
    window._run_button.click()
    thread_ref = window._thread
    assert thread_ref is not None
    _wait_settled(window, wait_for_signal, thread_ref.finished)
    return window


def test_completed_job_shows_its_output_path_in_the_table_and_status(
    qapp, wait_for_signal
):
    window = _completed_window(qapp, wait_for_signal, "/out/job-1.h5")

    assert window._table_jobs[0].output_path == "/out/job-1.h5"
    assert "/out/job-1.h5" in window._status_label.text()


def test_job_row_shows_dash_while_there_is_no_finished_time_yet(qapp):
    from giecar_seismic.ui.main_window import FINISHED_AT_COLUMN

    window = MainWindow()
    window._insert_job_row(0, Job(dataset_id=1, cutoff_hz=30.0, order=4, id=7))

    assert window._jobs_table.item(0, FINISHED_AT_COLUMN).text() == "-"


def test_open_output_button_enables_only_for_a_selected_job_with_an_output(
    qapp, wait_for_signal
):
    window = _completed_window(qapp, wait_for_signal, "/out/job-1.h5")
    window._insert_job_row(
        1, Job(dataset_id=1, cutoff_hz=30.0, order=4, id=99)
    )  # no output

    assert window._open_output_button.isEnabled() is False
    window._jobs_table.selectRow(0)
    assert window._open_output_button.isEnabled() is True
    window._jobs_table.selectRow(1)
    assert window._open_output_button.isEnabled() is False


def test_open_output_button_reveals_the_selected_jobs_output_folder(
    qapp, wait_for_signal, monkeypatch
):
    window = _completed_window(qapp, wait_for_signal, "/out/job-1.h5")
    revealed: list[str] = []
    monkeypatch.setattr(window, "_reveal_in_file_manager", revealed.append)

    window._jobs_table.selectRow(0)
    window._open_output_button.click()

    assert revealed == ["/out/job-1.h5"]


def test_main_window_still_builds_no_output_paths_itself():
    # the UI only displays/opens paths the service already put on the Job.
    source = inspect.getsource(main_window_module)
    assert ".h5" not in source


# --- View Output -> seismic viewer -------------------------------------------


def test_view_output_button_enables_only_for_a_selected_completed_job(
    qapp, wait_for_signal
):
    from PyQt5.QtWidgets import QWidget

    opened: list[int] = []
    window = _completed_window(qapp, wait_for_signal, "/out/job-1.h5")

    def fake_open_viewer(job_id: int, parent: QWidget) -> QWidget:
        opened.append(job_id)
        return QWidget(parent)

    window._open_viewer = fake_open_viewer
    window._insert_job_row(
        1,
        Job(
            dataset_id=1,
            cutoff_hz=30.0,
            order=4,
            id=2,
            status=JobStatus.CANCELLED,
            output_path="/out/job-2.h5",
        ),
    )

    assert window._view_output_button.isEnabled() is False  # nothing selected
    window._jobs_table.selectRow(1)  # CANCELLED: not viewable (even with output)
    assert window._view_output_button.isEnabled() is False
    window._jobs_table.selectRow(0)  # COMPLETED
    assert window._view_output_button.isEnabled() is True

    window._view_output_button.click()

    assert opened == [1]


def test_view_output_is_disabled_when_no_viewer_is_composed(qapp, wait_for_signal):
    window = _completed_window(qapp, wait_for_signal, "/out/job-1.h5")
    assert window._open_viewer is None

    window._jobs_table.selectRow(0)

    assert window._view_output_button.isEnabled() is False


# --- Preview -> seismic viewer with a FilterPreview (no job created) ----------


def _preview_window(qapp, opened: list):
    from PyQt5.QtWidgets import QWidget

    dataset = _dataset()
    service = _build_service(
        dataset, FakeTraceReader(np.zeros((4, N_SAMPLES))), FakeTraceWriter()
    )

    def fake_open_viewer(target, parent: QWidget) -> QWidget:
        opened.append(target)
        return QWidget(parent)

    window = MainWindow(service=service, open_viewer=fake_open_viewer)
    return window, dataset


def test_preview_button_needs_a_persisted_dataset_and_a_composed_viewer(qapp):
    opened: list = []
    window, dataset = _preview_window(qapp, opened)
    assert window._preview_button.isEnabled() is False  # no dataset yet

    window.set_dataset(dataset)
    assert window._preview_button.isEnabled() is True

    window._open_viewer = None
    window._refresh_controls()
    assert window._preview_button.isEnabled() is False


def test_preview_opens_the_viewer_with_the_current_parameters_and_no_job(
    qapp, wait_for_signal
):
    opened: list = []
    window, dataset = _preview_window(qapp, opened)
    window.set_dataset(dataset)
    window._filter_type_combo.setCurrentIndex(
        window._filter_type_combo.findData(FilterType.BAND_PASS)
    )
    window._upper_cutoff_spinbox.setValue(60.0)
    window._cutoff_spinbox.setValue(20.0)
    window._order_spinbox.setValue(6)

    window._preview_button.click()

    assert opened == [
        FilterPreview(
            dataset_id=1,
            cutoff_hz=20.0,
            order=6,
            filter_type=FilterType.BAND_PASS,
            upper_cutoff_hz=60.0,
        )
    ]
    assert window._jobs_table.rowCount() == 0  # nothing created, nothing run
    assert window._thread is None
    while window._history_thread is not None:
        wait_for_signal(window._history_thread.finished)


def test_preview_reports_parameters_rejected_by_the_viewer_without_opening_it(
    qapp, wait_for_signal
):
    opened: list = []
    window, dataset = _preview_window(qapp, opened)
    window.set_dataset(dataset)

    def rejecting_open_viewer(target, parent):
        raise InvalidFilterParametersError("boom")

    window._open_viewer = rejecting_open_viewer
    window._preview_button.click()

    assert window._status_label.text() == "Invalid filter: boom"
    assert window._viewer is None
    while window._history_thread is not None:
        wait_for_signal(window._history_thread.finished)


def test_job_log_dialog_can_be_maximized_and_minimized(qapp):
    dialog = JobLogDialog()
    flags = dialog.windowFlags()

    assert flags & Qt.WindowMaximizeButtonHint
    assert flags & Qt.WindowMinimizeButtonHint
    assert not flags & Qt.WindowContextHelpButtonHint
