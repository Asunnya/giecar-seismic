from pathlib import Path

from PyQt5.QtCore import QThread
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
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
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset
from giecar_seismic.ui.workers import DatasetImporter, FilterJobWorker, SegyImportWorker

JOBS_TABLE_HEADERS = ["id", "cutoff (Hz)", "order", "status"]

# Keeps the cutoff spinbox strictly below Nyquist (0 < cutoff_hz <
# nyquist_hz, per FilterJobService.create_filter_job) by one step.
CUTOFF_EPSILON_HZ = 0.01


class MainWindow(QMainWindow):
    """First vertical slice of the desktop UI: Dataset -> Filter ->
    Processing -> Jobs, wired to run a single filter job on a QThread.

    `service` is accepted by construction rather than composed here: the
    SQLAlchemy-backed repositories and the SEG-Y-backed reader/writer
    factories this needs for a real end-to-end run don't exist yet. With
    `service=None` (the entrypoint's current default) the window still
    opens and displays correctly, but "Run Filter" stays disabled -- an
    explicit, visible incomplete composition rather than a fake demo.

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
        dataset_importer: DatasetImporter = import_segy_dataset,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._dataset_importer = dataset_importer
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

        self.setWindowTitle("GIECAR Seismic Filter")
        self._build_ui()
        self._refresh_controls()

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

        self._cutoff_spinbox = QDoubleSpinBox(group)
        self._cutoff_spinbox.setDecimals(2)
        self._cutoff_spinbox.setRange(0.01, 1_000_000.0)
        self._cutoff_spinbox.setValue(30.0)
        form.addRow("Cutoff (Hz):", self._cutoff_spinbox)

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

        self._jobs_table = QTableWidget(0, len(JOBS_TABLE_HEADERS), group)
        self._jobs_table.setHorizontalHeaderLabels(JOBS_TABLE_HEADERS)
        self._jobs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._jobs_table)

        return group

    # -- Dataset wiring: Select SEG-Y -> QThread -> import_segy_dataset --

    def _on_select_segy_clicked(self) -> None:
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
        max_cutoff = max(CUTOFF_EPSILON_HZ, dataset.nyquist_hz - CUTOFF_EPSILON_HZ)
        self._cutoff_spinbox.setMaximum(max_cutoff)
        if self._cutoff_spinbox.value() > max_cutoff:
            self._cutoff_spinbox.setValue(max_cutoff)

        self._refresh_controls()

    # -- Run / Cancel ----------------------------------------------------

    def _can_run(self) -> bool:
        # dataset.id is None until the dataset is actually persisted
        # (SQLAlchemy integration lands later) -- create_filter_job()
        # needs a real dataset_id, so a not-yet-persisted dataset must
        # never make Run available. No id is invented here.
        return (
            self._service is not None
            and self._dataset is not None
            and self._dataset.id is not None
        )

    def _refresh_controls(self) -> None:
        running = self._thread is not None
        importing = self._import_thread is not None
        self._run_button.setEnabled(self._can_run() and not running and not importing)
        self._cancel_button.setEnabled(running)
        self._select_segy_button.setEnabled(not importing and not running)

    def _on_run_clicked(self) -> None:
        if not self._can_run() or self._thread is not None:
            return
        service = self._service
        dataset = self._dataset
        assert service is not None
        assert dataset is not None
        assert dataset.id is not None

        job = service.create_filter_job(
            dataset_id=dataset.id,
            cutoff_hz=self._cutoff_spinbox.value(),
            order=self._order_spinbox.value(),
        )
        self._add_job_row(job)
        assert job.id is not None
        self._current_job_id = job.id

        cancel_token = CooperativeCancelToken()
        thread = QThread(self)
        worker = FilterJobWorker(service, job.id, cancel_token)
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

        self._status_label.setText("Running...")
        self._progress_bar.setValue(0)
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
        self._finish_job("Completed", job)

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
        if self._thread is None and self._import_thread is None:
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

    # -- Jobs table -------------------------------------------------------

    def _add_job_row(self, job: Job) -> None:
        row = self._jobs_table.rowCount()
        self._jobs_table.insertRow(row)
        self._set_job_row(row, job)

    def _update_job_row(self, job: Job) -> None:
        for row in range(self._jobs_table.rowCount()):
            item = self._jobs_table.item(row, 0)
            if item is not None and item.text() == str(job.id):
                self._set_job_row(row, job)
                return

    def _set_job_row(self, row: int, job: Job) -> None:
        values = [str(job.id), str(job.cutoff_hz), str(job.order), job.status.name]
        for column, value in enumerate(values):
            self._jobs_table.setItem(row, column, QTableWidgetItem(value))
