"""2D seismic section viewer (QC of a filter job's output).

Layout: a navigation/display bar on top; below it, a splitter with the
section view on the left and, on the right, the selected trace's
metadata, an original-vs-filtered trace overlay and both amplitude
spectra with the job's cutoff marked.

Rendering is delegated to a SeismicRenderer (Matplotlib). This dialog
keeps all state -- orientation, line, display mode, gain, clip, colormap,
wiggle, selected trace and its spectrum -- and the worker lifecycle; a
renderer only draws what it is handed.

Threading: every load (geometry index build, section read) runs on a
QThread via the workers in viewer_workers.py; this dialog only receives
small value objects through signals and draws them. Navigation is
disabled while a load is in flight -- the simplest policy that makes a
stale result impossible. Trace selection and the spectrum are computed
from the section already in memory (one line, no I/O), so they run on
the GUI thread.
"""

import logging

import numpy as np
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from giecar_seismic.application.seismic_viewer import (
    SeismicSection,
    SeismicViewerService,
    TraceSpectrum,
    TraceView,
)
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.ui.matplotlib_renderer import MatplotlibSeismicRenderer
from giecar_seismic.ui.seismic_renderer import (
    COLORMAPS,
    DISPLAY_MODES,
    DisplaySettings,
    SeismicRenderer,
)
from giecar_seismic.ui.viewer_workers import (
    GeometryIndexBuilder,
    GeometryIndexWorker,
    SectionLoadWorker,
)

log = logging.getLogger(__name__)


# --- the dialog ----------------------------------------------------------------


class SeismicViewer(QDialog):
    def __init__(
        self,
        service: SeismicViewerService,
        build_geometry_index: GeometryIndexBuilder,
        job_id: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._build_geometry_index = build_geometry_index
        self._job_id = job_id
        self._context = service.context(job_id)  # small repository reads only
        self._line_numbers: list[int] = []
        self._section: SeismicSection | None = None
        self._selected: TraceView | None = None
        self._selected_spectrum: TraceSpectrum | None = None
        self._renderer: SeismicRenderer | None = None
        self._section_was_clicked = False
        # A line requested while a thread is still winding down (e.g. the
        # first line right after the geometry index finished) starts once
        # that thread's `finished` fires -- never two threads at once.
        self._pending_line: int | None = None

        # One worker/thread pair at a time, kept as attributes while alive.
        self._thread: QThread | None = None
        self._worker: GeometryIndexWorker | SectionLoadWorker | None = None

        self.setWindowTitle(
            f"Seismic viewer -- job {job_id} ({self._context.dataset.name}, "
            f"cutoff {self._context.job.cutoff_hz} Hz, order {self._context.job.order})"
        )
        self.resize(1400, 800)
        self._build_ui()
        self._start_geometry_index()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self._orientation_combo = QComboBox(self)
        self._orientation_combo.addItems(
            [o.value.capitalize() for o in LineOrientation]
        )
        self._orientation_combo.currentIndexChanged.connect(
            self._on_orientation_changed
        )
        self._prev_button = QPushButton("◀", self)
        self._prev_button.clicked.connect(lambda: self._step_line(-1))
        self._line_spinbox = QSpinBox(self)
        self._line_spinbox.setKeyboardTracking(False)
        self._line_spinbox.valueChanged.connect(self._on_line_spinbox_changed)
        self._next_button = QPushButton("▶", self)
        self._next_button.clicked.connect(lambda: self._step_line(1))
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItems(DISPLAY_MODES)
        self._mode_combo.currentIndexChanged.connect(self._redraw)
        self._wiggle_checkbox = QCheckBox("Wiggle", self)
        self._wiggle_checkbox.toggled.connect(self._redraw)
        self._gain_spinbox = QDoubleSpinBox(self)
        self._gain_spinbox.setRange(0.1, 100.0)
        self._gain_spinbox.setValue(1.0)
        self._gain_spinbox.setSingleStep(0.5)
        self._gain_spinbox.valueChanged.connect(self._redraw)
        self._clip_spinbox = QDoubleSpinBox(self)
        self._clip_spinbox.setRange(50.0, 100.0)
        self._clip_spinbox.setValue(99.0)
        self._clip_spinbox.valueChanged.connect(self._redraw)
        self._cmap_combo = QComboBox(self)
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.currentIndexChanged.connect(self._redraw)
        for label, widget in (
            ("Line:", self._orientation_combo),
            ("", self._prev_button),
            ("", self._line_spinbox),
            ("", self._next_button),
            ("Mode:", self._mode_combo),
            ("", self._wiggle_checkbox),
            ("Gain:", self._gain_spinbox),
            ("Clip %:", self._clip_spinbox),
            ("Colormap:", self._cmap_combo),
        ):
            if label:
                bar.addWidget(QLabel(label, self))
            bar.addWidget(widget)
        bar.addStretch(1)
        root.addLayout(bar)

        self._status_label = QLabel("", self)
        root.addWidget(self._status_label)

        self._splitter = QSplitter(self)  # horizontal by default

        self._trace_panel = QWidget(self._splitter)
        trace_layout = QVBoxLayout(self._trace_panel)
        self._trace_info_label = QLabel(
            "Click a trace on the section.", self._trace_panel
        )
        self._trace_info_label.setWordWrap(True)
        trace_layout.addWidget(self._trace_info_label)
        self._analysis_slot = QVBoxLayout()
        trace_layout.addLayout(self._analysis_slot, 1)

        self._install_renderer(MatplotlibSeismicRenderer())
        splitter = self._splitter
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self._set_navigation_enabled(False)

    # -- loading ---------------------------------------------------------------

    @property
    def orientation(self) -> LineOrientation:
        return list(LineOrientation)[self._orientation_combo.currentIndex()]

    def _set_navigation_enabled(self, enabled: bool) -> None:
        for widget in (
            self._orientation_combo,
            self._prev_button,
            self._line_spinbox,
            self._next_button,
        ):
            widget.setEnabled(enabled)

    def _start_worker(self, worker: GeometryIndexWorker | SectionLoadWorker) -> None:
        assert self._thread is None, "a load is already in flight"
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        for terminal in (
            (worker.finished, worker.failed)
            if isinstance(worker, GeometryIndexWorker)
            else (worker.loaded, worker.failed)
        ):
            terminal.connect(thread.quit)
            terminal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self._set_navigation_enabled(False)
        thread.start()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        if self._pending_line is not None:
            line, self._pending_line = self._pending_line, None
            self._request_line(line)
            return
        if self._line_numbers:
            self._set_navigation_enabled(True)

    def _start_geometry_index(self) -> None:
        self._status_label.setText("Indexing seismic geometry...")
        worker = GeometryIndexWorker(self._build_geometry_index, self._context.dataset)
        worker.finished.connect(self._on_geometry_indexed)
        worker.failed.connect(self._on_load_failed)
        self._start_worker(worker)

    def _on_geometry_indexed(self, _built: bool) -> None:
        dataset_id = self._context.dataset.id
        assert dataset_id is not None
        self._line_numbers = self._service.line_numbers(dataset_id, self.orientation)
        if not self._line_numbers:
            self._status_label.setText("No geometry found for this dataset.")
            return
        self._line_spinbox.blockSignals(True)
        self._line_spinbox.setRange(self._line_numbers[0], self._line_numbers[-1])
        self._line_spinbox.setValue(self._line_numbers[0])
        self._line_spinbox.blockSignals(False)
        self._pending_line = self._line_numbers[0]
        if self._thread is None:  # orientation switch: no thread winding down
            self._on_thread_finished()

    def _on_orientation_changed(self, _index: int) -> None:
        if self._thread is not None:
            return
        self._line_numbers = []
        self._on_geometry_indexed(False)

    def _step_line(self, direction: int) -> None:
        if not self._line_numbers or self._thread is not None:
            return
        current = self._line_spinbox.value()
        position = int(np.searchsorted(self._line_numbers, current))
        position = min(max(position + direction, 0), len(self._line_numbers) - 1)
        self._line_spinbox.setValue(self._line_numbers[position])

    def _on_line_spinbox_changed(self, value: int) -> None:
        if not self._line_numbers or self._thread is not None:
            return
        # snap to the nearest existing line -- not every number in the
        # spinbox range exists on an irregular survey.
        position = int(np.argmin(np.abs(np.asarray(self._line_numbers) - value)))
        nearest = self._line_numbers[position]
        if nearest != value:
            self._line_spinbox.blockSignals(True)
            self._line_spinbox.setValue(nearest)
            self._line_spinbox.blockSignals(False)
        self._request_line(nearest)

    def _request_line(self, line_number: int) -> None:
        self._status_label.setText(f"Loading {self.orientation.value} {line_number}...")
        worker = SectionLoadWorker(
            self._service, self._job_id, self.orientation, line_number
        )
        worker.loaded.connect(self._on_section_loaded)
        worker.failed.connect(self._on_load_failed)
        self._start_worker(worker)

    def _on_section_loaded(self, section: SeismicSection) -> None:
        self._section = section
        self._selected = None
        self._selected_spectrum = None
        self._section_was_clicked = False
        self._status_label.setText(
            f"{section.orientation.value.capitalize()} {section.line_number}: "
            f"{section.n_present_traces} traces on a {len(section.coordinates)}-position axis, "
            f"{section.n_samples} samples"
        )
        self._redraw()
        self._show_selected_trace()

    def _on_load_failed(self, message: str) -> None:
        self._status_label.setText(f"Failed: {message}")

    # -- renderer lifecycle ------------------------------------------------------

    def _install_renderer(self, renderer: SeismicRenderer) -> None:
        """Place a renderer's widgets: section on the left of the splitter,
        trace/spectrum under the info label on the right. Only one renderer
        exists at a time -- the previous one is disposed first."""
        if self._renderer is not None:
            self._renderer.coordinate_clicked.disconnect(self.select_coordinate)
            self._renderer.dispose()
        self._renderer = renderer
        renderer.coordinate_clicked.connect(self.select_coordinate)
        self._splitter.insertWidget(0, renderer.section_widget())
        self._analysis_slot.addWidget(renderer.analysis_widget())
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)

    @property
    def renderer(self) -> SeismicRenderer:
        assert self._renderer is not None
        return self._renderer

    # -- drawing (GUI thread, from the in-memory section only) --------------------

    def display_settings(self) -> DisplaySettings:
        return DisplaySettings(
            mode=self._mode_combo.currentText(),
            wiggle=self._wiggle_checkbox.isChecked(),
            gain=self._gain_spinbox.value(),
            clip_percentile=self._clip_spinbox.value(),
            colormap=self._cmap_combo.currentText(),
        )

    def _redraw(self, *_args: object) -> None:
        self.renderer.show_section(self._section, self.display_settings())
        if self._section is not None:
            log.debug(
                "%s rendered %s %s in %.1f ms",
                type(self.renderer).__name__,
                self._section.orientation.value,
                self._section.line_number,
                self.renderer.last_render_seconds * 1000,
            )

    def select_coordinate(self, coordinate: float) -> None:
        if self._section is None:
            return
        self._section_was_clicked = True
        view = self._service.select_trace(self._job_id, self._section, coordinate)
        self._selected = view
        self._selected_spectrum = (
            self._service.spectrum(view) if view is not None else None
        )
        self._show_selected_trace()

    def _show_selected_trace(self) -> None:
        view = self._selected
        if view is None:
            self._trace_info_label.setText(
                "No trace at this position (missing in the survey footprint)."
                if self._section is not None and self._section_was_clicked
                else "Click a trace on the section."
            )
        else:
            g, d, j = view.geometry, view.dataset, view.job
            self._trace_info_label.setText(
                f"Trace {g.trace_index}  |  inline {g.inline}, crossline {g.crossline}\n"
                f"{d.n_samples} samples @ {d.sample_rate_ms} ms  (Nyquist {d.nyquist_hz:.1f} Hz)\n"
                f"Job {j.id}: cutoff {j.cutoff_hz} Hz, order {j.order}"
            )
        self.renderer.show_trace(view, self._selected_spectrum)

    # -- shutdown ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent | None) -> None:
        # Same policy as MainWindow: never destroy a running QThread. A
        # section load is short; the user simply closes again afterwards.
        if event is None:
            return
        if self._thread is not None:
            self._status_label.setText(
                "Waiting for the current load to finish before closing..."
            )
            event.ignore()
            return
        if self._renderer is not None:
            self._renderer.coordinate_clicked.disconnect(self.select_coordinate)
            self._renderer.dispose()
            self._renderer = None
        event.accept()
