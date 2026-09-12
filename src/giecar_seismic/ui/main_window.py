from collections.abc import Callable
from datetime import datetime
from math import ceil
from pathlib import Path

from PyQt5.QtCore import QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QDesktopServices
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
    InvalidFilterParametersError,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import FilterType, Job, JobStatus
from giecar_seismic.ui.filter_labels import FILTER_LABELS, FILTER_NAMES, cutoff_summary
from giecar_seismic.ui.workers import (
    DatasetImporter,
    FilterJobWorker,
    JobHistoryWorker,
    SegyImportWorker,
    dataset_labels,
)

JOBS_TABLE_HEADERS = [
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
DATASET_COLUMN = JOBS_TABLE_HEADERS.index("Dataset")
STATUS_COLUMN = JOBS_TABLE_HEADERS.index("Status")
PROGRESS_COLUMN = JOBS_TABLE_HEADERS.index("Progress")
CREATED_AT_COLUMN = JOBS_TABLE_HEADERS.index("Created at")
FINISHED_AT_COLUMN = JOBS_TABLE_HEADERS.index("Finished at")
ALL_DATASETS = "All datasets"
ALL_STATUSES = "All statuses"


def format_datetime(value: datetime | None) -> str:
    """Stable, readable timestamp for the history table. Datetimes are naive
    local time throughout the domain; no timezone is invented here."""
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else "-"


def format_progress(progress: float) -> str:
    return f"{progress:g}%"


# Opens the seismic viewer for a job id; composed in __main__ so this
# window never imports the viewer's service/infrastructure wiring.
ViewerOpener = Callable[[int, QWidget], QWidget]

# Keeps the cutoff spinbox strictly below Nyquist (0 < cutoff_hz <
# nyquist_hz, per FilterJobService.create_filter_job) by one step.
CUTOFF_EPSILON_HZ = 0.01


class MainWindow(QMainWindow):
    # Emitted on the GUI thread once a history query has finished -- table
    # rebuilt from the list_jobs() result and its QThread already gone.
    history_refreshed = pyqtSignal()

    """First vertical slice of the desktop UI: Dataset -> Filter ->
    Processing -> Jobs, wired to run a single filter job on a QThread.

    Dependencies are accepted by construction rather than composed here.
    `dataset_importer` is the full import (read SEG-Y metadata + persist
    to SQLite -- application.dataset_import.ImportDatasetUseCase in
    production), so the dataset that arrives via set_dataset() already
    carries its persistent id; this window never touches a repository or
    SQLAlchemy itself. `service` is still not composed by the entrypoint
    (the SEG-Y/HDF5 reader/writer factories it needs for a real run
    aren't wired yet), so "Run Filter" stays disabled -- an explicit,
    visible incomplete composition rather than a fake demo. Either
    dependency left as None disables the corresponding control.

    Selecting a SEG-Y file records its path and starts a background
    import (see SegyImportWorker) that reads trace headers and produces a
    SeismicDataset, which then flows into set_dataset(). That import
    thread/worker is tracked separately from the filter job's -- they are
    two distinct, independently-lived worker pairs (`_import_thread`/
    `_import_worker` vs `_thread`/`_worker`), never conflated, since a
    dataset import and a filter run are different operations that can, in
    principle, be in flight independently.
    """

    def __init__(
        self,
        service: FilterJobService | None = None,
        dataset_importer: DatasetImporter | None = None,
        open_viewer: ViewerOpener | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._dataset_importer = dataset_importer
        self._open_viewer = open_viewer
        self._viewer: QWidget | None = None
        self._dataset: SeismicDataset | None = None
        self._current_job_id: int | None = None

        # Kept as instance attributes -- not local variables -- for the
        # whole lifetime of a run, so the QThread/worker are never
        # collected while the thread is still alive. Cleared only once
        # QThread.finished fires (see _on_thread_finished).
        self._thread: QThread | None = None
        self._worker: FilterJobWorker | None = None

        # Same pattern, kept entirely separate: a SEG-Y header import in
        # flight must never be confused with a filter job in flight.
        self._import_thread: QThread | None = None
        self._import_worker: SegyImportWorker | None = None

        # History query, same pattern. The visible rows are a display of
        # the last list_jobs() result (`_table_jobs`, row-aligned); SQLite
        # is the source of truth -- a terminal job state triggers a
        # refresh to reconcile with what was persisted.
        self._history_thread: QThread | None = None
        self._history_worker: JobHistoryWorker | None = None
        self._table_jobs: list[Job] = []
        self._dataset_trace_counts: dict[int, int] = {}
        self._dataset_labels: dict[int, str] = {}  # dataset id -> file name

        self.setWindowTitle("GIECAR Seismic Filter")
        self._build_ui()
        self._update_filter_controls()
        self._refresh_controls()
        self._history_loaded_once = False
        self._history_result_pending = False
        if self._service is not None:
            # First load once the event loop is running -- never a
            # synchronous query inside widget construction. A no-op if
            # something already refreshed the history before it fires.
            QTimer.singleShot(0, self._initial_history_load)

    # -- UI construction -----------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_dataset_group())
        layout.addWidget(self._build_filter_group())
        layout.addWidget(self._build_processing_group())
        layout.addWidget(self._build_jobs_group())
        self.setCentralWidget(central)

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Dataset", self)
        form = QFormLayout(group)

        path_row = QHBoxLayout()
        self._segy_path_edit = QLineEdit(group)
        self._segy_path_edit.setReadOnly(True)
        self._select_segy_button = QPushButton("Select SEG-Y", group)
        self._select_segy_button.clicked.connect(self._on_select_segy_clicked)
        path_row.addWidget(self._segy_path_edit)
        path_row.addWidget(self._select_segy_button)
        form.addRow("SEG-Y path:", path_row)

        self._name_label = QLabel("-", group)
        self._inlines_label = QLabel("-", group)
        self._crosslines_label = QLabel("-", group)
        self._traces_label = QLabel("-", group)
        self._samples_label = QLabel("-", group)
        self._sample_rate_label = QLabel("-", group)
        self._nyquist_label = QLabel("-", group)
        form.addRow("Name:", self._name_label)
        form.addRow("Inlines:", self._inlines_label)
        form.addRow("Crosslines:", self._crosslines_label)
        form.addRow("Traces:", self._traces_label)
        form.addRow("Samples:", self._samples_label)
        form.addRow("Sample rate:", self._sample_rate_label)
        form.addRow("Nyquist:", self._nyquist_label)

        return group

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Filter", self)
        form = QFormLayout(group)

        self._filter_type_combo = QComboBox(group)
        for kind in FilterType:
            self._filter_type_combo.addItem(FILTER_LABELS[kind], kind)
        form.addRow("Filter type:", self._filter_type_combo)

        self._cutoff_spinbox = QDoubleSpinBox(group)
        self._cutoff_spinbox.setDecimals(2)
        self._cutoff_spinbox.setRange(0.01, 1_000_000.0)
        self._cutoff_spinbox.setValue(30.0)
        self._cutoff_label = QLabel("High cutoff (Hz):", group)
        form.addRow(self._cutoff_label, self._cutoff_spinbox)
        self._upper_cutoff_spinbox = QDoubleSpinBox(group)
        self._upper_cutoff_spinbox.setDecimals(2)
        self._upper_cutoff_spinbox.setRange(0.02, 1_000_000.0)
        self._upper_cutoff_spinbox.setValue(40.0)
        self._upper_cutoff_label = QLabel("High cutoff (Hz):", group)
        form.addRow(self._upper_cutoff_label, self._upper_cutoff_spinbox)
        self._band_hint = QLabel(
            "Band-pass requires low < high < Nyquist. Raise high first to raise low.",
            group,
        )
        self._band_hint.setWordWrap(True)
        form.addRow(self._band_hint)
        self._filter_type_combo.currentIndexChanged.connect(
            self._update_filter_controls
        )
        self._cutoff_spinbox.valueChanged.connect(self._update_filter_controls)
        self._upper_cutoff_spinbox.valueChanged.connect(self._update_filter_controls)

        self._order_spinbox = QSpinBox(group)
        self._order_spinbox.setRange(2, 8)
        self._order_spinbox.setValue(4)
        form.addRow("Order:", self._order_spinbox)

        self._run_button = QPushButton("Run Filter", group)
        self._run_button.clicked.connect(self._on_run_clicked)
        form.addRow(self._run_button)

        return group

    def _build_processing_group(self) -> QGroupBox:
        group = QGroupBox("Processing", self)
        layout = QVBoxLayout(group)

        self._status_label = QLabel("Idle", group)
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar(group)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._cancel_button = QPushButton("Cancel", group)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self._cancel_button)

        return group

    def _build_jobs_group(self) -> QGroupBox:
        group = QGroupBox("Jobs", self)
        layout = QVBoxLayout(group)

        filters = QHBoxLayout()
        self._dataset_filter = QComboBox(group)
        self._dataset_filter.addItem(ALL_DATASETS)
        self._dataset_filter.currentIndexChanged.connect(
            self._on_history_filter_changed
        )
        self._status_filter = QComboBox(group)
        self._status_filter.addItem(ALL_STATUSES)
        self._status_filter.addItems([status.name for status in JobStatus])
        self._status_filter.currentIndexChanged.connect(self._on_history_filter_changed)
        self._refresh_button = QPushButton("Refresh", group)
        self._refresh_button.clicked.connect(self.refresh_history)
        self._history_status_label = QLabel("", group)
        for widget in (
            QLabel("Dataset:", group),
            self._dataset_filter,
            QLabel("Status:", group),
            self._status_filter,
            self._refresh_button,
            self._history_status_label,
        ):
            filters.addWidget(widget)
        filters.addStretch(1)
        layout.addLayout(filters)

        self._jobs_table = QTableWidget(0, len(JOBS_TABLE_HEADERS), group)
        self._jobs_table.setHorizontalHeaderLabels(JOBS_TABLE_HEADERS)
        self._jobs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._jobs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._jobs_table.itemSelectionChanged.connect(self._refresh_output_buttons)
        layout.addWidget(self._jobs_table)

        # Reveals the selected job's output in the OS file manager. The
        # path itself always comes from the Job (set by the service); the
        # UI never builds output locations.
        self._open_output_button = QPushButton("Open output folder", group)
        self._open_output_button.setEnabled(False)
        self._open_output_button.clicked.connect(self._on_open_output_clicked)
        layout.addWidget(self._open_output_button)

        self._resume_button = QPushButton("Resume", group)
        self._resume_button.setEnabled(False)
        self._resume_button.clicked.connect(self._on_resume_clicked)
        layout.addWidget(self._resume_button)

        self._view_output_button = QPushButton("View Output", group)
        self._view_output_button.setEnabled(False)
        self._view_output_button.clicked.connect(self._on_view_output_clicked)
        layout.addWidget(self._view_output_button)

        return group

    # -- Dataset wiring: Select SEG-Y -> QThread -> dataset importer ------

    def _on_select_segy_clicked(self) -> None:
        if self._dataset_importer is None:
            return  # no importer composed: the button is disabled too
        if self._import_thread is not None or self._thread is not None:
            # An import is already running, or a filter job is: ignore
            # (the button is disabled in both cases too). Never let a new
            # import replace the dataset a currently-running job is using.
            return

        path = self._prompt_for_segy_path()
        if not path:
            return  # user cancelled the dialog: do nothing

        self._segy_path_edit.setText(path)
        self._start_dataset_import(path)

    def _prompt_for_segy_path(self) -> str:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Select SEG-Y file", "", "SEG-Y files (*.segy *.sgy);;All files (*)"
        )
        return path

    def _start_dataset_import(self, path: str) -> None:
        # Invalidate whatever dataset is currently selected the instant a
        # new import starts: the path shown must never end up paired with
        # metadata/dataset belonging to a different, earlier file --
        # including if this new import goes on to fail.
        self._dataset = None
        self._clear_dataset_labels()

        assert self._dataset_importer is not None
        name = Path(path).stem
        thread = QThread(self)
        worker = SegyImportWorker(path, name, importer=self._dataset_importer)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_dataset_imported)
        worker.failed.connect(self._on_dataset_import_failed)

        # Same lifecycle pattern as the filter worker below: deleteLater()
        # on the worker is connected to the worker's *own* terminal
        # signals (so it is requested while the worker's thread event
        # loop is still running to actually process it), quit()/
        # deleteLater() on the thread itself only via thread.finished.
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_import_thread_finished)

        self._import_thread = thread
        self._import_worker = worker

        self._status_label.setText("Loading dataset...")
        self._refresh_controls()

        thread.start()

    def _on_dataset_imported(self, dataset: SeismicDataset) -> None:
        self.set_dataset(dataset)
        self._status_label.setText("Dataset loaded")

    def _on_dataset_import_failed(self, message: str) -> None:
        # Nothing to roll back: _start_dataset_import() already invalidated
        # the previous dataset and cleared its labels, so a failed import
        # leaves no dataset (None) rather than a stale or partial one.
        self._status_label.setText(f"Failed to load dataset: {message}")

    def _on_import_thread_finished(self) -> None:
        self._import_thread = None
        self._import_worker = None
        self._refresh_controls()

    def _clear_dataset_labels(self) -> None:
        for label in (
            self._name_label,
            self._inlines_label,
            self._crosslines_label,
            self._traces_label,
            self._samples_label,
            self._sample_rate_label,
            self._nyquist_label,
        ):
            label.setText("-")

    def set_dataset(self, dataset: SeismicDataset) -> None:
        self._dataset = dataset
        self._name_label.setText(dataset.name)
        self._inlines_label.setText(str(dataset.n_inlines))
        self._crosslines_label.setText(str(dataset.n_crosslines))
        self._traces_label.setText(str(dataset.n_traces))
        self._samples_label.setText(str(dataset.n_samples))
        self._sample_rate_label.setText(f"{dataset.sample_rate_ms} ms")
        self._nyquist_label.setText(f"{dataset.nyquist_hz:.2f} Hz")

        # UI-level guardrail mirroring the domain rule 0 < cutoff_hz <
        # nyquist_hz -- FilterJobService.create_filter_job remains the
        # actual enforcement point; this only steers the widget away from
        # an obviously invalid value before it ever reaches the service.
        self._update_filter_controls()

        self._refresh_controls()

    @property
    def filter_type(self) -> FilterType:
        return self._filter_type_combo.currentData()

    def _update_filter_controls(self, *_args: object) -> None:
        band = self.filter_type is FilterType.BAND_PASS
        self._cutoff_label.setText(
            "High cutoff (Hz):"
            if self.filter_type is FilterType.LOW_PASS
            else "Low cutoff (Hz):"
        )
        self._upper_cutoff_spinbox.setVisible(band)
        self._upper_cutoff_label.setVisible(band)
        self._band_hint.setVisible(band)
        # Largest representable hundredth strictly below Nyquist, even
        # when Nyquist itself is not a multiple of the widget precision.
        maximum = (
            (ceil(self._dataset.nyquist_hz * 100) - 1) / 100
            if self._dataset is not None
            else 1_000_000.0
        )
        low, high = self._cutoff_spinbox, self._upper_cutoff_spinbox
        low.blockSignals(True)
        high.blockSignals(True)
        try:
            low.setRange(CUTOFF_EPSILON_HZ, max(CUTOFF_EPSILON_HZ, maximum))
            high.setRange(CUTOFF_EPSILON_HZ, max(CUTOFF_EPSILON_HZ, maximum))
            if band and maximum >= 2 * CUTOFF_EPSILON_HZ:
                low.setMaximum(maximum - CUTOFF_EPSILON_HZ)
                high.setMinimum(low.value() + CUTOFF_EPSILON_HZ)
                low.setMaximum(high.value() - CUTOFF_EPSILON_HZ)
        finally:
            low.blockSignals(False)
            high.blockSignals(False)
        self._refresh_controls()

    # -- Run / Cancel ----------------------------------------------------

    def _can_run(self) -> bool:
        # dataset.id is None for a dataset that was never persisted (e.g.
        # one handed to set_dataset() directly) -- create_filter_job()
        # needs a real dataset_id, so such a dataset must never make Run
        # available. No id is invented here.
        return (
            self._service is not None
            and self._dataset is not None
            and self._dataset.id is not None
            and 0 < self._cutoff_spinbox.value() < self._dataset.nyquist_hz
            and (
                self.filter_type is not FilterType.BAND_PASS
                or self._cutoff_spinbox.value()
                < self._upper_cutoff_spinbox.value()
                < self._dataset.nyquist_hz
            )
        )

    def _refresh_controls(self) -> None:
        running = self._thread is not None
        importing = self._import_thread is not None
        self._run_button.setEnabled(self._can_run() and not running and not importing)
        self._cancel_button.setEnabled(running)
        self._refresh_output_buttons()
        self._select_segy_button.setEnabled(
            self._dataset_importer is not None and not importing and not running
        )

    def _on_run_clicked(self) -> None:
        if not self._can_run() or self._thread is not None:
            return
        service = self._service
        dataset = self._dataset
        assert service is not None
        assert dataset is not None
        assert dataset.id is not None

        try:
            job = service.create_filter_job(
                dataset_id=dataset.id,
                cutoff_hz=self._cutoff_spinbox.value(),
                order=self._order_spinbox.value(),
                filter_type=self.filter_type,
                upper_cutoff_hz=(
                    self._upper_cutoff_spinbox.value()
                    if self.filter_type is FilterType.BAND_PASS
                    else None
                ),
            )
        except InvalidFilterParametersError as exc:
            self._status_label.setText(f"Invalid filter: {exc}")
            return
        self._register_session_dataset(dataset)
        self._insert_job_row(0, job)  # newest first, like the persisted history
        self._start_job_worker(job, resume=False)

    def _on_resume_clicked(self) -> None:
        if not self._can_resume():
            return
        job = self._selected_job()
        assert job is not None
        self._start_job_worker(job, resume=True)

    def _start_job_worker(self, job: Job, *, resume: bool) -> None:
        service = self._service
        assert service is not None and job.id is not None
        self._current_job_id = job.id

        cancel_token = CooperativeCancelToken()
        thread = QThread(self)
        worker = FilterJobWorker(service, job.id, cancel_token, resume=resume)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.completed.connect(self._on_job_completed)
        worker.failed.connect(self._on_job_failed)
        worker.cancelled.connect(self._on_job_cancelled)

        # Cooperative shutdown: whichever terminal signal fires asks the
        # thread's event loop to quit -- this file never forcibly ends
        # the OS thread itself. worker.deleteLater() is connected to the
        # worker's *own* terminal signals (not to thread.finished): that
        # way the deferred-delete request is posted to the worker's own
        # thread queue while that thread's event loop is still running to
        # actually process it, instead of after the thread has already
        # stopped pumping events. thread.deleteLater() is connected to
        # thread.finished, since by then we're back on the GUI thread
        # (the QThread object's own affinity) and it's safe there.
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker

        self._status_label.setText("Resuming..." if resume else "Running...")
        self._progress_bar.setValue(round(job.progress) if resume else 0)
        self._refresh_controls()

        thread.start()

    def _on_cancel_clicked(self) -> None:
        if self._service is None or self._current_job_id is None:
            return
        try:
            # Cooperative cancellation only: this asks the running worker
            # to stop at its next safe chunk boundary. The QThread itself
            # is never touched here -- it keeps running until the worker
            # observes the request and one of the terminal signals fires.
            self._service.cancel_job(self._current_job_id)
        except Exception:  # noqa: BLE001, S110 -- a benign race (job already finished) must not crash the GUI
            pass

    # -- Signal handlers (run on the GUI thread) ------------------------

    def _on_progress(self, percent: float) -> None:
        self._progress_bar.setValue(int(percent))

    def _on_job_completed(self, job: Job) -> None:
        self._progress_bar.setValue(100)
        self._finish_job(f"Completed: {job.output_path}", job)

    def _on_job_cancelled(self, job: Job) -> None:
        self._finish_job("Cancelled", job)

    def _on_job_failed(self, message: str) -> None:
        self._finish_job(f"Failed: {message}", None)

    def _finish_job(self, status_text: str, job: Job | None) -> None:
        self._status_label.setText(status_text)
        resolved_job = job
        if (
            resolved_job is None
            and self._service is not None
            and self._current_job_id is not None
        ):
            resolved_job = self._service.get_job_status(self._current_job_id)
        if resolved_job is not None:
            self._update_job_row(resolved_job)
        self._current_job_id = None
        # Reconcile the screen with what was actually persisted.
        self.refresh_history()

    def _on_thread_finished(self) -> None:
        # deleteLater() for both the worker and the thread was already
        # requested via their own signals (see _on_run_clicked) -- this
        # only drops MainWindow's Python-level references, once the
        # thread has genuinely finished.
        self._worker = None
        self._thread = None
        self._refresh_controls()

    # -- Window shutdown --------------------------------------------------

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Never destroy a running QThread.

        Policy (deliberately the simplest one that is still safe, not a
        shutdown framework): if either the filter or the import thread is
        still alive, the close is rejected (event.ignore()) instead of
        letting Qt tear down a QMainWindow whose QThread children are
        still running -- which would destroy a live thread. For the
        filter job, which does have a cooperative cancellation mechanism,
        cancellation is requested so the thread reaches a safe stopping
        point sooner; the SEG-Y header import has no such token yet, so
        closing during an import just means waiting for it to finish
        naturally. Either way, nothing here blocks the GUI thread with an
        unbounded thread.wait(), and the OS thread is never forcibly
        ended.
        """
        if event is None:
            return
        if (
            self._thread is None
            and self._import_thread is None
            and self._history_thread is None
        ):
            event.accept()
            return

        if self._current_job_id is not None and self._service is not None:
            try:
                self._service.cancel_job(self._current_job_id)
            except Exception:  # noqa: BLE001, S110 -- best-effort; closing must not crash on a benign race
                pass

        self._status_label.setText(
            "Waiting for background work to finish before closing..."
        )
        event.ignore()

    # -- Jobs history ------------------------------------------------------

    def _selected_history_filters(self) -> tuple[int | None, JobStatus | None]:
        data = self._dataset_filter.currentData()
        dataset_id = None if data is None else int(data)
        status_text = self._status_filter.currentText()
        status = None if status_text == ALL_STATUSES else JobStatus[status_text]
        return dataset_id, status

    def _on_history_filter_changed(self, _index: int) -> None:
        self.refresh_history()

    def _set_history_controls_enabled(self, enabled: bool) -> None:
        for widget in (self._dataset_filter, self._status_filter, self._refresh_button):
            widget.setEnabled(enabled)

    def _initial_history_load(self) -> None:
        if not self._history_loaded_once:
            self.refresh_history()

    def refresh_history(self) -> None:
        """Query list_jobs(dataset_id, status) off the GUI thread and rebuild
        the table from the result. One query at a time: filters and Refresh
        are disabled while it runs."""
        if self._service is None or self._history_thread is not None:
            return
        dataset_id, status = self._selected_history_filters()
        thread = QThread(self)
        worker = JobHistoryWorker(self._service, dataset_id, status)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.trace_counts_ready.connect(self._on_trace_counts_loaded)
        worker.succeeded.connect(self._on_history_loaded)
        worker.failed.connect(self._on_history_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_history_thread_finished)
        self._history_thread = thread
        self._history_worker = worker
        self._history_loaded_once = True
        self._set_history_controls_enabled(False)
        thread.start()

    def _on_trace_counts_loaded(self, counts: dict[int, int]) -> None:
        self._dataset_trace_counts.update(counts)

    def _on_history_loaded(self, jobs: list[Job], labels: dict[int, str]) -> None:
        # Newest first: created_at, then id as a deterministic tie-break.
        ordered = sorted(jobs, key=lambda j: (j.created_at, j.id or 0), reverse=True)
        self._dataset_labels.update(labels)
        self._rebuild_dataset_filter()
        self._jobs_table.setRowCount(0)
        self._table_jobs = []
        for job in ordered:
            self._insert_job_row(self._jobs_table.rowCount(), job)
        self._history_status_label.setText(f"{len(ordered)} jobs")
        self._refresh_output_buttons()
        self._history_result_pending = True

    def _on_history_failed(self, message: str) -> None:
        self._history_status_label.setText(f"History unavailable: {message}")

    def _on_history_thread_finished(self) -> None:
        self._history_thread = None
        self._history_worker = None
        self._set_history_controls_enabled(True)
        if self._history_result_pending:
            self._history_result_pending = False
            self.history_refreshed.emit()

    def _register_session_dataset(self, dataset: SeismicDataset) -> None:
        """Label the current dataset before its first job row appears; the
        next history load replaces this with the worker's labels."""
        assert dataset.id is not None
        self._dataset_trace_counts[dataset.id] = dataset.n_traces
        if dataset.id in self._dataset_labels:
            return
        label = dataset_labels({dataset.id: dataset})[dataset.id]
        if label in self._dataset_labels.values():
            label = f"{label} (#{dataset.id})"
        self._dataset_labels[dataset.id] = label
        self._rebuild_dataset_filter()

    def _dataset_label(self, dataset_id: int) -> str:
        return self._dataset_labels.get(dataset_id, f"Dataset #{dataset_id}")

    def _rebuild_dataset_filter(self) -> None:
        # (label, id) items; the id travels as item data so a label is
        # never parsed back into an id.
        wanted: list[tuple[str, int | None]] = [(ALL_DATASETS, None)] + [
            (self._dataset_labels[i], i) for i in sorted(self._dataset_labels)
        ]
        current = self._dataset_filter.currentData()
        existing = [
            (self._dataset_filter.itemText(i), self._dataset_filter.itemData(i))
            for i in range(self._dataset_filter.count())
        ]
        if existing == wanted:
            return
        self._dataset_filter.blockSignals(True)
        self._dataset_filter.clear()
        for label, dataset_id in wanted:
            self._dataset_filter.addItem(label, dataset_id)
        self._dataset_filter.setCurrentIndex(
            max(0, self._dataset_filter.findData(current)) if current is not None else 0
        )
        self._dataset_filter.blockSignals(False)

    # -- Jobs table rows --------------------------------------------------------

    def _insert_job_row(self, row: int, job: Job) -> None:
        self._jobs_table.insertRow(row)
        self._table_jobs.insert(row, job)
        self._set_job_row(row, job)

    def _update_job_row(self, job: Job) -> None:
        for row, existing in enumerate(self._table_jobs):
            if existing.id == job.id:
                self._table_jobs[row] = job
                self._set_job_row(row, job)
                return

    def _set_job_row(self, row: int, job: Job) -> None:
        values = [
            str(job.id),
            self._dataset_label(job.dataset_id),
            FILTER_NAMES[job.filter_type],
            cutoff_summary(job),
            str(job.order),
            job.status.name,
            format_progress(job.progress),
            format_datetime(job.created_at),
            format_datetime(job.finished_at),
        ]
        for column, value in enumerate(values):
            self._jobs_table.setItem(row, column, QTableWidgetItem(value))
        self._refresh_output_buttons()

    def _selected_job(self) -> Job | None:
        rows = {index.row() for index in self._jobs_table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._table_jobs[row] if row < len(self._table_jobs) else None

    def _selected_output_path(self) -> str | None:
        job = self._selected_job()
        return job.output_path if job is not None else None

    def _can_resume(self) -> bool:
        job = self._selected_job()
        return (
            self._service is not None
            and self._thread is None
            and self._import_thread is None
            and job is not None
            and job.status is JobStatus.CANCELLED
            and job.output_path is not None
            and 0
            <= job.processed_traces
            < self._dataset_trace_counts.get(job.dataset_id, 0)
            and Path(job.output_path).is_file()
        )

    def _refresh_output_buttons(self) -> None:
        self._resume_button.setEnabled(self._can_resume())
        self._open_output_button.setEnabled(self._selected_output_path() is not None)
        self._view_output_button.setEnabled(
            self._selected_viewable_job_id() is not None
        )

    def _selected_viewable_job_id(self) -> int | None:
        # Only COMPLETED jobs (a finalized HDF5) are viewable -- including
        # ones loaded from history: the viewer resolves the job's own
        # dataset by job.dataset_id, never this window's current dataset.
        job = self._selected_job()
        if self._open_viewer is None or job is None or job.output_path is None:
            return None
        if job.status is not JobStatus.COMPLETED:
            return None
        return job.id

    def _on_view_output_clicked(self) -> None:
        job_id = self._selected_viewable_job_id()
        if job_id is None or self._open_viewer is None:
            return
        self._viewer = self._open_viewer(job_id, self)
        self._viewer.show()

    def _on_open_output_clicked(self) -> None:
        path = self._selected_output_path()
        if path is not None:
            self._reveal_in_file_manager(path)

    def _reveal_in_file_manager(self, path: str) -> None:
        # Opens the containing folder with the platform's file manager;
        # extracted so tests can observe the call without launching one.
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
