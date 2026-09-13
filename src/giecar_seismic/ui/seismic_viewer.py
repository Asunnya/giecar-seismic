"""2D seismic section viewer: navigation by inline/crossline plus
REGION-oriented spectral QC of a filter job's output (or a preview).

Layout: a navigation/display bar on top; below it, a splitter with the
section view on the left and, on the right, the compact Region QC panel:
the region's bounds/counts, spectrum controls and the regional spectrum
(aggregate original vs aggregate filtered).

Interaction: one normal click on the section analyses that single trace
(a region of width one); a second click extends it to the contiguous
region between the two clicks (any order; both inclusive). Every
physically present trace inside it contributes; gaps are excluded and
reported. A third click starts over from a new single trace. No keyboard
modifier, no drag.

Rendering is delegated to a SeismicRenderer (PyQtGraph by default,
Matplotlib selectable at runtime) for the section and a SpectrumRenderer
of the same family for the spectrum. This dialog keeps all state --
orientation, line, display mode, gain, clip, colormap, wiggle, region
boundaries, the raw regional spectrum and its display variant -- and the
worker lifecycle; renderers only draw what they are handed. Switching
renderer disposes the old ones, builds new ones and redraws what is
already in memory: no worker, no SEG-Y/HDF5 read, no repository query,
no FFT.

Threading: every load (geometry index build, section read) and the
regional spectrum aggregation (chunked FFTs over possibly hundreds of
traces) run on a QThread via the workers in viewer_workers.py -- one
worker/thread pair at a time; this dialog only receives small value
objects through signals and draws them. Navigation and region clicks
are disabled while a worker is in flight -- the simplest policy that
makes a stale result impossible. Presentation changes (scale, curve
visibility, renderer) reuse the cached regional spectrum on the GUI
thread.
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
    MIN_REGION_TRACES,
    RegionalSpectrum,
    ResolvedRegion,
    SectionRegion,
    SeismicSection,
    SeismicViewerService,
    SpectrumScale,
    ViewerTarget,
    regional_spectrum_for_display,
)
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.ui.filter_labels import FILTER_NAMES, cutoff_summary
from giecar_seismic.ui.matplotlib_renderer import MatplotlibSeismicRenderer
from giecar_seismic.ui.pyqtgraph_renderer import PyQtGraphSeismicRenderer
from giecar_seismic.ui.region_spectrum_window import (
    RegionSpectrumWindow,
    axis_abbreviation,
    make_spectrum_renderer,
    region_summary,
)
from giecar_seismic.ui.seismic_renderer import (
    COLORMAPS,
    DISPLAY_MODES,
    RENDERERS,
    DisplaySettings,
    SeismicRenderer,
)
from giecar_seismic.ui.spectrum_renderer import SpectrumRenderer, SpectrumVisibility
from giecar_seismic.ui.viewer_workers import (
    GeometryIndexBuilder,
    GeometryIndexWorker,
    RegionalSpectrumWorker,
    SectionLoadWorker,
)

log = logging.getLogger(__name__)

REGION_PROMPT = (
    "Region QC\nClick a trace on the section for its spectrum; click a second "
    "position to analyse the region between them."
)
EXTEND_HINT = "Click a second boundary to extend the region."


def make_renderer(name: str) -> SeismicRenderer:
    if name == "PyQtGraph":
        return PyQtGraphSeismicRenderer()
    return MatplotlibSeismicRenderer()


# --- the dialog ----------------------------------------------------------------


class SeismicViewer(QDialog):
    def __init__(
        self,
        service: SeismicViewerService,
        build_geometry_index: GeometryIndexBuilder,
        target: ViewerTarget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._build_geometry_index = build_geometry_index
        self._target = target
        self._context = service.context(target)  # small repository reads only
        self._line_numbers: list[int] = []
        self._section: SeismicSection | None = None
        # Region QC state (viewer-owned; renderers only draw it):
        # NO_BOUNDARY -> ONE_BOUNDARY(start) -> COMPLETE(region) -> ONE_BOUNDARY...
        self._region_start: int | None = None
        self._region: ResolvedRegion | None = None
        self._regional: RegionalSpectrum | None = None  # raw, linear (cached)
        self._display: RegionalSpectrum | None = None  # scale/visibility applied
        self._region_request_id = 0
        self._pending_region_request: int | None = None
        # A click while a regional worker is still running (typically the
        # second click right after the first) is applied once that thread
        # finishes -- never two workers at once, never a lost click.
        self._pending_click: float | None = None
        self._renderer: SeismicRenderer | None = None
        self._spectrum_renderer: SpectrumRenderer | None = None
        self._region_window: RegionSpectrumWindow | None = None
        self._section_request_id = 0
        self._pending_section_request: int | None = None
        # A line requested while a thread is still winding down (e.g. the
        # first line right after the geometry index finished) starts once
        # that thread's `finished` fires -- never two threads at once.
        self._pending_line: int | None = None

        # One worker/thread pair at a time, kept as attributes while alive.
        self._thread: QThread | None = None
        self._worker: (
            GeometryIndexWorker | SectionLoadWorker | RegionalSpectrumWorker | None
        ) = None

        job = self._context.job
        subject = (
            "preview (nothing persisted)" if self._context.preview else f"job {job.id}"
        )
        self.setWindowTitle(
            f"Seismic viewer -- {subject} ({self._context.dataset.name}, "
            f"{FILTER_NAMES[job.filter_type]} {cutoff_summary(job)}, "
            f"order {job.order})"
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
        # View Output exists to show the job's result, so it opens on the
        # filtered section; a Preview is about inspecting the raw data
        # first, so it opens on the original. Set before connecting: no
        # renderer is installed yet, so _redraw must not fire here.
        self._mode_combo.setCurrentText(
            "Original" if self._context.preview else "Filtered"
        )
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
        self._renderer_combo = QComboBox(self)
        self._renderer_combo.addItems(RENDERERS)  # default: first entry
        self._renderer_combo.currentIndexChanged.connect(self._on_renderer_changed)
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
            ("Renderer:", self._renderer_combo),
        ):
            if label:
                bar.addWidget(QLabel(label, self))
            bar.addWidget(widget)
        bar.addStretch(1)
        root.addLayout(bar)

        self._status_label = QLabel("", self)
        root.addWidget(self._status_label)

        self._splitter = QSplitter(self)  # horizontal by default

        self._region_panel = QWidget(self._splitter)
        region_layout = QVBoxLayout(self._region_panel)
        self._region_label = QLabel(REGION_PROMPT, self._region_panel)
        self._region_label.setWordWrap(True)
        region_layout.addWidget(self._region_label)
        spectrum_bar = QHBoxLayout()
        self._spectrum_scale_combo = QComboBox(self)
        self._spectrum_scale_combo.addItems([scale.value for scale in SpectrumScale])
        self._spectrum_scale_combo.setToolTip(
            "dB uses the aggregate original peak as a common reference for both "
            "curves; floor -120 dB. Applied after the linear regional mean."
        )
        self._spectrum_scale_combo.currentIndexChanged.connect(
            self._on_spectrum_settings_changed
        )
        spectrum_bar.addWidget(QLabel("Spectrum scale:", self))
        spectrum_bar.addWidget(self._spectrum_scale_combo)
        region_layout.addLayout(spectrum_bar)
        self._original_checkbox = QCheckBox("Aggregate original", self)
        self._filtered_checkbox = QCheckBox("Aggregate filtered", self)
        self._response_checkbox = QCheckBox("Show filter response", self)
        self._response_checkbox.setToolTip(
            "Ideal zero-phase gain |H|² from the processing SOS, on the separate right axis."
        )
        for checkbox in (
            self._original_checkbox,
            self._filtered_checkbox,
        ):
            checkbox.setChecked(True)
        for checkbox in (
            self._original_checkbox,
            self._filtered_checkbox,
            self._response_checkbox,
        ):
            checkbox.toggled.connect(self._on_spectrum_settings_changed)
            region_layout.addWidget(checkbox)
        self._open_spectrum_button = QPushButton("Open Spectrum", self)
        self._open_spectrum_button.setToolTip(
            "Larger, modeless view of the current regional spectrum."
        )
        self._open_spectrum_button.setEnabled(False)
        self._open_spectrum_button.clicked.connect(self._open_region_window)
        region_layout.addWidget(self._open_spectrum_button)
        self._spectrum_slot = QVBoxLayout()
        region_layout.addLayout(self._spectrum_slot, 1)

        self._install_renderer(self._renderer_combo.currentText())
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

    def _start_worker(
        self, worker: GeometryIndexWorker | SectionLoadWorker | RegionalSpectrumWorker
    ) -> None:
        assert self._thread is None, "a worker is already in flight"
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        for name in worker.terminal_signal_names:
            terminal = getattr(worker, name)
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
        if self._pending_click is not None:
            click, self._pending_click = self._pending_click, None
            self.select_coordinate(click)

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
        self._section_request_id += 1
        request_id = self._section_request_id
        self._pending_section_request = request_id
        self._clear_region()
        self._status_label.setText(f"Loading {self.orientation.value} {line_number}...")
        worker = SectionLoadWorker(
            self._service, self._target, self.orientation, line_number
        )
        worker.loaded.connect(
            lambda section: self._accept_section_result(request_id, section)
        )
        worker.failed.connect(self._on_load_failed)
        self._start_worker(worker)

    def _accept_section_result(self, request_id: int, section: SeismicSection) -> None:
        # Single in-flight load is still the policy. Consume each result once;
        # an obsolete/duplicate delivery cannot clear a newer trace selection.
        if request_id != self._pending_section_request:
            return
        self._pending_section_request = None
        self._on_section_loaded(section)

    def _on_section_loaded(self, section: SeismicSection) -> None:
        self._section = section
        self._clear_region()
        self._status_label.setText(
            f"{section.orientation.value.capitalize()} {section.line_number}: "
            f"{section.n_present_traces} traces on a {len(section.coordinates)}-position axis, "
            f"{section.n_samples} samples"
        )
        self._redraw()

    def _on_load_failed(self, message: str) -> None:
        self._pending_section_request = None
        self._pending_region_request = None
        self._clear_region()
        self._status_label.setText(f"Failed: {message}")

    # -- renderer lifecycle ------------------------------------------------------

    def _install_renderer(self, name: str) -> None:
        """Place a section renderer's widget on the left of the splitter and
        a spectrum renderer of the same family under the region panel. Only
        one of each exists at a time -- the previous ones are disposed."""
        if self._renderer is not None:
            self._renderer.coordinate_clicked.disconnect(self.select_coordinate)
            self._renderer.dispose()
        if self._spectrum_renderer is not None:
            self._spectrum_slot.removeWidget(self._spectrum_renderer.widget())
            self._spectrum_renderer.dispose()
        renderer = make_renderer(name)
        self._renderer = renderer
        renderer.coordinate_clicked.connect(self.select_coordinate)
        self._splitter.insertWidget(0, renderer.section_widget())
        self._spectrum_renderer = make_spectrum_renderer(name)
        self._spectrum_slot.addWidget(self._spectrum_renderer.widget())
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)

    def _on_renderer_changed(self, _index: int) -> None:
        # Presentation only: the section, region and regional spectrum are
        # already in memory and are handed to the new renderers as-is.
        # Viewport (zoom/pan) is reset; nothing else changes.
        self._renderer_combo.setEnabled(False)
        try:
            self._install_renderer(self._renderer_combo.currentText())
            self._redraw()
            self._show_display()
        finally:
            self._renderer_combo.setEnabled(True)

    @property
    def renderer(self) -> SeismicRenderer:
        assert self._renderer is not None
        return self._renderer

    @property
    def spectrum_renderer(self) -> SpectrumRenderer:
        assert self._spectrum_renderer is not None
        return self._spectrum_renderer

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
        self._draw_region()
        if self._section is not None:
            log.debug(
                "%s rendered %s %s in %.1f ms",
                type(self.renderer).__name__,
                self._section.orientation.value,
                self._section.line_number,
                self.renderer.last_render_seconds * 1000,
            )

    # -- region QC (two clicks; section in memory only) ----------------------------

    @property
    def region(self) -> ResolvedRegion | None:
        return self._region

    @property
    def regional_spectrum(self) -> RegionalSpectrum | None:
        """The cached raw (linear) regional spectrum, if computed."""
        return self._regional

    def select_coordinate(self, coordinate: float) -> None:
        """A normal click on the section. The first click analyses that
        single trace (region of width one) and keeps the boundary; the
        second click completes the region between the two (either order);
        a click after a complete region starts over. Clicks are snapped
        to the section's axis, never to the nearest physical trace: a
        boundary may sit on a gap. A click during the (short) regional
        worker is queued and applied when it finishes."""
        section = self._section
        if section is None:
            return
        if self._thread is not None:
            if isinstance(self._worker, RegionalSpectrumWorker):
                self._pending_click = coordinate
            return
        position = section.position_for_coordinate(coordinate)
        if position is None:
            return  # outside the axis
        clicked = int(section.coordinates[position])
        if self._region_start is None:
            # NO_BOUNDARY or COMPLETE -> ONE_BOUNDARY(clicked): single trace
            self._region_start = clicked
            region = SectionRegion(clicked, clicked)
            hint = EXTEND_HINT
        else:
            region = SectionRegion.from_boundaries(self._region_start, clicked)
            self._region_start = None
            hint = None
        self._region = region.resolve(section)
        self._set_regional(None)
        self._draw_region()
        summary = region_summary(self._region)
        if hint is not None:
            summary += f"\n{hint}"
        if not self._region.is_valid:
            self._region_label.setText(
                f"{summary}\nRegion must contain at least {MIN_REGION_TRACES} trace."
            )
            return
        self._region_label.setText(f"{summary}\nCalculating regional spectrum...")
        self._region_request_id += 1
        request_id = self._region_request_id
        self._pending_region_request = request_id
        worker = RegionalSpectrumWorker(
            self._service, section, region, self._context.dataset, self._context.job
        )
        worker.computed.connect(
            lambda regional: self._accept_regional_result(request_id, regional)
        )
        worker.failed.connect(
            lambda message: self._on_regional_failed(request_id, message)
        )
        self._start_worker(worker)

    def _region_text(self) -> str:
        assert self._region is not None
        text = region_summary(self._region)
        if self._region_start is not None:
            text += f"\n{EXTEND_HINT}"
        return text

    def _accept_regional_result(
        self, request_id: int, regional: RegionalSpectrum
    ) -> None:
        # A result for a region/section that is no longer current is dropped.
        if request_id != self._pending_region_request or self._region is None:
            return
        self._pending_region_request = None
        self._region_label.setText(self._region_text())
        self._set_regional(regional)

    def _on_regional_failed(self, request_id: int, message: str) -> None:
        if request_id != self._pending_region_request:
            return
        self._pending_region_request = None
        self._set_regional(None)
        if self._region is not None:
            self._region_label.setText(
                f"{self._region_text()}\nRegional spectrum failed: {message}"
            )

    def _axis_abbreviation(self) -> str:
        return axis_abbreviation(self.orientation)

    def _clear_region(self) -> None:
        """Section replaced/cleared or navigation requested: no boundary,
        no region, no regional spectrum, empty window."""
        self._region_start = None
        self._region = None
        self._pending_region_request = None
        self._pending_click = None
        self._set_regional(None)
        self._region_label.setText(REGION_PROMPT)
        if self._renderer is not None:
            self._renderer.clear_region()

    def _draw_region(self) -> None:
        """Re-apply the viewer-owned region state to the section renderer."""
        renderer = self._renderer
        if renderer is None:
            return
        if self._region is not None:
            bounds = self._region.region
            renderer.show_region(bounds.lower_coordinate, bounds.upper_coordinate)
        elif self._region_start is not None:
            renderer.show_region_start(self._region_start)
        else:
            renderer.clear_region()

    def _set_regional(self, regional: RegionalSpectrum | None) -> None:
        self._regional = regional
        self._refresh_spectrum()

    def _spectrum_visibility(self) -> SpectrumVisibility:
        return SpectrumVisibility(
            self._original_checkbox.isChecked(),
            self._filtered_checkbox.isChecked(),
            self._response_checkbox.isChecked(),
        )

    def _on_spectrum_settings_changed(self, *_args: object) -> None:
        self._refresh_spectrum()

    def _refresh_spectrum(self) -> None:
        """Presentation only: transform the cached raw regional spectrum
        (no FFT) and hand it to the spectrum renderer and the window."""
        visibility = self._spectrum_visibility()
        self._display = (
            regional_spectrum_for_display(
                self._regional,
                SpectrumScale(self._spectrum_scale_combo.currentText()),
                show_filter_response=visibility.response,
            )
            if self._regional is not None
            else None
        )
        self._show_display()
        self._open_spectrum_button.setEnabled(self._regional is not None)
        if self._region_window is not None:
            self._region_window.set_regional(self._regional)

    def _show_display(self) -> None:
        """Hand the already-transformed display spectrum to the renderer."""
        if self._spectrum_renderer is not None:
            self._spectrum_renderer.show_waveform(
                self._regional.waveform if self._regional is not None else None
            )
            self._spectrum_renderer.show_spectrum(
                self._display.spectrum if self._display is not None else None,
                self._spectrum_visibility(),
            )

    def _open_region_window(self) -> None:
        if self._regional is None:
            return
        if self._region_window is None:
            window = RegionSpectrumWindow(
                self._regional,
                self,
                renderer_name=self._renderer_combo.currentText(),
                scale=SpectrumScale(self._spectrum_scale_combo.currentText()),
            )
            self._region_window = window
            window.finished.connect(self._on_region_window_closed)
        self._region_window.show()
        self._region_window.raise_()
        self._region_window.activateWindow()

    def _on_region_window_closed(self, _result: int) -> None:
        window = self._region_window
        if window is not None:
            window.finished.disconnect(self._on_region_window_closed)
            self._region_window = None

    # -- shutdown ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent | None) -> None:
        # Same policy as MainWindow: never destroy a running QThread. A
        # section load or regional aggregation is short; the user simply
        # closes again afterwards.
        if event is None:
            return
        if self._thread is not None:
            self._status_label.setText(
                "Waiting for the current load to finish before closing..."
            )
            event.ignore()
            return
        if self._region_window is not None:
            self._region_window.close()
        if self._renderer is not None:
            self._renderer.coordinate_clicked.disconnect(self.select_coordinate)
            self._renderer.dispose()
            self._renderer = None
        if self._spectrum_renderer is not None:
            self._spectrum_renderer.dispose()
            self._spectrum_renderer = None
        event.accept()

    def reject(self) -> None:
        # Escape follows the same worker guard and child-window cleanup as X.
        self.close()
