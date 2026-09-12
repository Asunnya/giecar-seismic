"""Renderer abstraction: Matplotlib and PyQtGraph renderers receive the very
same section/trace/spectrum objects, agree on what they draw (shape,
extent, coordinates, levels), and switching between them in the viewer
is presentation-only -- no service call, no worker, no I/O, state kept.
"""

import inspect
import threading
from dataclasses import replace

import numpy as np
import pytest

from giecar_seismic.application.seismic_viewer import (
    SeismicSection,
    TraceSpectrum,
    TraceView,
    ViewerContext,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation, TraceGeometry
from giecar_seismic.domain.job import Job, JobStatus
from giecar_seismic.ui import matplotlib_renderer as mpl_module
from giecar_seismic.ui import pyqtgraph_renderer as pg_module
from giecar_seismic.ui import viewer_workers as workers_module
from giecar_seismic.ui.matplotlib_renderer import MatplotlibSeismicRenderer
from giecar_seismic.ui.pyqtgraph_renderer import PyQtGraphSeismicRenderer
from giecar_seismic.ui.seismic_renderer import (
    COLORMAPS,
    DISPLAY_MODES,
    RENDERERS,
    DisplaySettings,
    amplitude_limit,
    panels_for_mode,
    section_extent,
)
from giecar_seismic.ui.seismic_viewer import SeismicViewer

N_SAMPLES = 32
INLINES = [10, 11, 12]
CROSSLINES = [1, 2, 3, 4]


def _dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="s",
        source_path="/s.segy",
        n_inlines=3,
        n_crosslines=4,
        n_traces=11,
        n_samples=N_SAMPLES,
        sample_rate_ms=4.0,
    )


def _job() -> Job:
    return Job(
        id=7,
        dataset_id=1,
        cutoff_hz=30.0,
        order=4,
        status=JobStatus.COMPLETED,
        output_path="/out/job-7.h5",
        progress=100.0,
    )


def _section(orientation=LineOrientation.INLINE, line=10) -> SeismicSection:
    axis = np.asarray(CROSSLINES if orientation is LineOrientation.INLINE else INLINES)
    physical = np.arange(len(axis)) + 10 * (line - 10)
    if line == 11:  # (11, 3) missing
        physical = np.asarray([13, 14, -1, 16])
    rng = np.random.default_rng(line)
    original = rng.standard_normal((len(axis), N_SAMPLES)).astype(np.float32)
    filtered = (original * 0.5).astype(np.float32)
    original[physical < 0] = np.nan
    filtered[physical < 0] = np.nan
    return SeismicSection(orientation, line, axis, physical, original, filtered, 4.0)


def _view(section: SeismicSection, position: int) -> TraceView:
    return TraceView(
        geometry=TraceGeometry(
            int(section.physical_trace_indices[position]),
            section.line_number,
            int(section.coordinates[position]),
        ),
        time_ms=section.time_ms,
        original=section.original[position],
        filtered=section.filtered[position],
        dataset=_dataset(),
        job=_job(),
    )


def _spectrum(view: TraceView) -> TraceSpectrum:
    f = np.fft.rfftfreq(N_SAMPLES, d=0.004)
    return TraceSpectrum(
        f,
        np.abs(np.fft.rfft(view.original)),
        np.abs(np.fft.rfft(view.filtered)),
        125.0,
        30.0,
    )


SETTINGS = DisplaySettings(
    mode="Original", wiggle=False, gain=1.5, clip_percentile=98.0, colormap="seismic"
)


@pytest.fixture(
    params=[MatplotlibSeismicRenderer, PyQtGraphSeismicRenderer], ids=["mpl", "pg"]
)
def renderer(request, qapp):
    r = request.param()
    yield r
    r.dispose()


# --- structural ----------------------------------------------------------------


def test_renderer_selector_offers_both_with_pyqtgraph_first():
    assert RENDERERS == ["PyQtGraph", "Matplotlib"]


def test_workers_import_no_plotting_library():
    lines = [
        l
        for l in inspect.getsource(workers_module).splitlines()
        if l.startswith(("import ", "from "))
    ]
    assert not any(
        "matplotlib" in l or "pyqtgraph" in l or "QtWidgets" in l for l in lines
    )


@pytest.mark.parametrize("module", [mpl_module, pg_module])
def test_renderers_do_no_io_and_know_no_infrastructure(module):
    lines = [
        l
        for l in inspect.getsource(module).splitlines()
        if l.startswith(("import ", "from "))
    ]
    assert not any(
        "infrastructure" in l
        or "segyio" in l
        or "h5py" in l
        or "sqlalchemy" in l
        or "viewer_workers" in l
        for l in lines
    )


# --- equivalence: both renderers draw the same thing -------------------------------


@pytest.mark.parametrize("mode", DISPLAY_MODES)
def test_both_renderers_record_identical_panels_for_the_same_section(qapp, mode):
    section = _section()
    settings = DisplaySettings(mode, False, 1.5, 98.0, "gray")
    mpl, pg = MatplotlibSeismicRenderer(), PyQtGraphSeismicRenderer()

    mpl.show_section(section, settings)
    pg.show_section(section, settings)

    assert mpl.last_panels == pg.last_panels
    assert len(mpl.last_panels) == (2 if mode == "Side-by-side" else 1)
    limit = amplitude_limit(section, 98.0, 1.5)
    for panel in mpl.last_panels:
        assert panel.shape == (len(CROSSLINES), N_SAMPLES)
        assert panel.extent == section_extent(section)
        assert panel.coordinates == tuple(CROSSLINES)
        assert panel.levels == (-limit, limit)
    assert [p.title for p in mpl.last_panels] == [
        t for t, _ in panels_for_mode(section, mode)
    ]
    mpl.dispose()
    pg.dispose()


def test_panels_share_the_sections_memory_not_copies():
    section = _section()
    for _title, data in panels_for_mode(section, "Original") + panels_for_mode(
        section, "Filtered"
    ):
        assert np.shares_memory(data, section.original) or np.shares_memory(
            data, section.filtered
        )


@pytest.mark.parametrize("mode", DISPLAY_MODES)
def test_each_renderer_draws_the_mode_with_time_downwards(renderer, mode):
    section = _section()
    renderer.show_section(section, DisplaySettings(mode, False, 1.0, 99.0, "seismic"))

    n_panels = 2 if mode == "Side-by-side" else 1
    if isinstance(renderer, MatplotlibSeismicRenderer):
        axes = renderer.section_figure.axes
        assert len(axes) == n_panels
        assert all(ax.get_ylim()[0] > ax.get_ylim()[1] for ax in axes)
        assert all(ax.get_xlabel() == "Crossline" for ax in axes)
        if n_panels == 2:
            assert axes[0].images[0].get_clim() == axes[1].images[0].get_clim()
    else:
        assert len(renderer.plots) == n_panels
        assert all(p.getViewBox().yInverted() for p in renderer.plots)
        assert len(renderer.images) == n_panels
        if n_panels == 2:
            assert (
                renderer.images[0].levels.tolist() == renderer.images[1].levels.tolist()
            )
            assert (
                renderer.plots[1].getViewBox().linkedView(0)
                is renderer.plots[0].getViewBox()
            )


def test_each_renderer_supports_wiggle(renderer):
    section = _section()
    renderer.show_section(
        section, DisplaySettings("Original", True, 1.0, 99.0, "seismic")
    )

    assert renderer.last_panels[0].wiggle is True
    if isinstance(renderer, MatplotlibSeismicRenderer):
        ax = renderer.section_figure.axes[0]
        assert len(ax.images) == 0 and len(ax.get_lines()) == len(CROSSLINES)
    else:
        assert renderer.images == []
        assert len(renderer.plots[0].listDataItems()) >= len(CROSSLINES)


def test_each_renderer_offers_every_colormap(renderer):
    section = _section()
    for cmap in COLORMAPS:
        renderer.show_section(
            section, DisplaySettings("Original", False, 1.0, 99.0, cmap)
        )
        assert renderer.last_panels


def test_gain_and_clip_only_change_levels_not_data(renderer):
    section = _section()
    renderer.show_section(
        section, DisplaySettings("Original", False, 1.0, 99.0, "gray")
    )
    before = renderer.last_panels[0]
    renderer.show_section(
        section, DisplaySettings("Original", False, 2.0, 90.0, "gray")
    )
    after = renderer.last_panels[0]

    assert after.levels != before.levels
    assert after.levels == (
        -amplitude_limit(section, 90.0, 2.0),
        amplitude_limit(section, 90.0, 2.0),
    )
    assert after.shape == before.shape and after.coordinates == before.coordinates


def test_each_renderer_shows_trace_overlay_and_spectrum_with_cutoff(renderer):
    section = _section()
    view = _view(section, 1)
    spectrum = _spectrum(view)
    renderer.show_section(section, SETTINGS)

    renderer.show_trace(view, spectrum)

    if isinstance(renderer, MatplotlibSeismicRenderer):
        trace_ax = renderer.trace_figure.axes[0]
        assert len(trace_ax.get_lines()) == 2
        assert trace_ax.get_xlim() == (-trace_ax.get_xlim()[1], trace_ax.get_xlim()[1])
        spectrum_ax = renderer.spectrum_figure.axes[0]
        assert spectrum_ax.get_xlim()[1] == pytest.approx(125.0)
        assert any(
            "cutoff 30.0" in line.get_label() for line in spectrum_ax.get_lines()
        )
    else:
        assert renderer.cutoff_hz == 30.0
        assert renderer._cutoff_line.value() == pytest.approx(30.0)
        assert renderer._spectrum_plot.getViewBox().viewRange()[0][1] == pytest.approx(
            125.0
        )
        assert len(renderer._trace_original.xData) == N_SAMPLES
        assert len(renderer._trace_filtered.xData) == N_SAMPLES

    renderer.show_trace(None, None)  # clears without error


def test_each_renderer_emits_the_clicked_coordinate_only(renderer):
    section = _section()
    renderer.show_section(section, SETTINGS)
    clicked: list[float] = []
    renderer.coordinate_clicked.connect(clicked.append)

    if isinstance(renderer, MatplotlibSeismicRenderer):

        class Event:
            xdata = 2.3

        renderer._on_click(Event())
    else:
        renderer.click_at_coordinate(2.3)

    assert clicked == [2.3]  # resolving to a physical trace is the viewer's job


# --- comparison set: Ctrl+click and markers ------------------------------------------


def test_ctrl_click_emits_only_the_compare_signal(renderer):
    renderer.show_section(_section(), SETTINGS)
    clicked: list[float] = []
    compared: list[float] = []
    renderer.coordinate_clicked.connect(clicked.append)
    renderer.compare_coordinate_clicked.connect(compared.append)

    if isinstance(renderer, MatplotlibSeismicRenderer):

        class Plain:
            xdata = 1.2

        class Ctrl:
            xdata = 2.3
            key = "control"

        renderer._on_click(Plain())
        renderer._on_click(Ctrl())
    else:
        renderer.click_at_coordinate(1.2)
        renderer.click_at_coordinate(2.3, compare=True)

    # keyboard semantics stay inside the renderer: the viewer only sees two signals
    assert clicked == [1.2]
    assert compared == [2.3]


def _marker_lines(renderer):
    """Library-specific count of drawn comparison markers (all panels)."""
    if isinstance(renderer, MatplotlibSeismicRenderer):
        return [
            line
            for ax in renderer.section_figure.axes
            for line in ax.lines
            if line.get_gid() == "compare-marker"
        ]
    import pyqtgraph as pg

    return [
        item
        for plot in renderer.plots
        for item in plot.items
        if isinstance(item, pg.InfiniteLine)
    ]


@pytest.mark.parametrize("mode", ["Original", "Side-by-side"])
def test_compare_markers_are_drawn_per_panel_and_cleared(renderer, mode):
    section = _section()
    renderer.show_section(section, replace(SETTINGS, mode=mode))
    n_panels = 2 if mode == "Side-by-side" else 1

    renderer.show_compare_markers([1.0, 3.0])

    assert renderer.last_compare_markers == (1.0, 3.0)
    lines = _marker_lines(renderer)
    assert len(lines) == 2 * n_panels
    xs = sorted(
        {
            float(line.get_xdata()[0])
            if isinstance(renderer, MatplotlibSeismicRenderer)
            else float(line.value())
            for line in lines
        }
    )
    assert xs == [1.0, 3.0]

    renderer.show_compare_markers([3.0])  # selection shrank: redrawn, not appended
    assert renderer.last_compare_markers == (3.0,)
    assert len(_marker_lines(renderer)) == n_panels

    renderer.show_compare_markers([])
    assert renderer.last_compare_markers == ()
    assert _marker_lines(renderer) == []

    renderer.show_compare_markers([2.0])
    renderer.show_section(section, replace(SETTINGS, mode=mode))  # a new section
    assert renderer.last_compare_markers == ()  # the viewer re-applies its own state
    assert _marker_lines(renderer) == []
    renderer.show_section(None, SETTINGS)
    assert _marker_lines(renderer) == []


def test_both_renderers_record_identical_compare_markers(qapp):
    mpl, pg = MatplotlibSeismicRenderer(), PyQtGraphSeismicRenderer()
    try:
        for r in (mpl, pg):
            r.show_section(_section(), SETTINGS)
            r.show_compare_markers([2.0, 4.0])
        assert mpl.last_compare_markers == pg.last_compare_markers == (2.0, 4.0)
    finally:
        mpl.dispose()
        pg.dispose()


# --- viewer integration: switching renderer ----------------------------------------


class FakeViewerService:
    def __init__(self):
        self.load_calls: list[tuple[LineOrientation, int]] = []
        self.select_calls = 0
        self.spectrum_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def context(self, job_id):
        return ViewerContext(dataset=_dataset(), job=_job())

    def line_numbers(self, dataset_id, orientation):
        return INLINES if orientation is LineOrientation.INLINE else CROSSLINES

    def load_section(self, job_id, orientation, line_number):
        self.load_calls.append((orientation, line_number))
        self.entered.set()
        assert self.release.wait(timeout=5)
        return _section(orientation, line_number)

    def select_trace(self, job_id, section, coordinate):
        self.select_calls += 1
        position = int(np.argmin(np.abs(section.coordinates - coordinate)))
        if section.physical_trace_indices[position] < 0:
            return None
        return _view(section, position)

    def spectrum(self, view):
        self.spectrum_calls += 1
        return _spectrum(view)


def _open(qapp, wait_for_signal):
    service = FakeViewerService()
    viewer = SeismicViewer(service, lambda d: True, target=7)
    for _ in range(2):
        thread = viewer._thread
        assert thread is not None
        wait_for_signal(thread.finished)
    return viewer, service


def test_viewer_defaults_to_pyqtgraph_and_offers_matplotlib(qapp, wait_for_signal):
    viewer, _ = _open(qapp, wait_for_signal)
    assert viewer._renderer_combo.currentText() == "PyQtGraph"
    assert [viewer._renderer_combo.itemText(i) for i in range(2)] == RENDERERS
    assert isinstance(viewer.renderer, PyQtGraphSeismicRenderer)


def test_switching_renderer_keeps_state_and_does_no_io_or_worker(qapp, wait_for_signal):
    viewer, service = _open(qapp, wait_for_signal)
    viewer._line_spinbox.setValue(11)
    wait_for_signal(viewer._thread.finished)
    viewer._mode_combo.setCurrentText("Difference")
    viewer._gain_spinbox.setValue(1.5)
    viewer._clip_spinbox.setValue(98.0)
    viewer._cmap_combo.setCurrentText("gray")
    viewer.select_coordinate(2.0)  # crossline 2 on inline 11 -> physical 14
    section_before = viewer._section
    selected_before = viewer._selected
    assert selected_before is not None and selected_before.geometry.trace_index == 14
    loads, selects, spectra = (
        len(service.load_calls),
        service.select_calls,
        service.spectrum_calls,
    )
    old_renderer = viewer.renderer

    viewer._renderer_combo.setCurrentText("Matplotlib")

    assert isinstance(viewer.renderer, MatplotlibSeismicRenderer)
    assert viewer.renderer is not old_renderer
    assert viewer._thread is None  # no worker started
    assert len(service.load_calls) == loads  # no reload
    assert (
        service.select_calls == selects and service.spectrum_calls == spectra
    )  # no re-resolve/FFT
    assert viewer._section is section_before  # same object, not a copy
    assert viewer.orientation is LineOrientation.INLINE
    assert viewer._line_spinbox.value() == 11
    assert viewer._mode_combo.currentText() == "Difference"
    assert viewer._gain_spinbox.value() == 1.5
    assert viewer._clip_spinbox.value() == 98.0
    assert viewer._cmap_combo.currentText() == "gray"
    assert viewer._selected is selected_before
    assert viewer._renderer_combo.isEnabled() is True
    # the new renderer drew the same section with the same settings
    panel = viewer.renderer.last_panels[0]
    assert panel.title.startswith("Filtered - Original -- inline 11")
    assert panel.levels == (
        -amplitude_limit(section_before, 98.0, 1.5),
        amplitude_limit(section_before, 98.0, 1.5),
    )
    assert any(
        "cutoff 30.0" in line.get_label()
        for line in viewer.renderer.spectrum_figure.axes[0].get_lines()
    )
    assert "Trace 14" in viewer._trace_info_label.text()


def test_trace_selection_resolves_the_same_physical_trace_in_both_renderers(
    qapp, wait_for_signal
):
    viewer, _ = _open(qapp, wait_for_signal)
    viewer._line_spinbox.setValue(11)
    wait_for_signal(viewer._thread.finished)

    results = {}
    for name in RENDERERS:
        viewer._renderer_combo.setCurrentText(name)
        viewer.renderer.coordinate_clicked.emit(1.2)  # via the renderer's own signal
        present = viewer._selected.geometry.trace_index
        viewer.renderer.coordinate_clicked.emit(3.0)  # (11, 3) missing
        missing = viewer._selected
        results[name] = (present, missing)

    assert results["Matplotlib"] == results["PyQtGraph"] == (13, None)
    assert "No trace at this position" in viewer._trace_info_label.text()


def test_switching_back_and_forth_does_not_duplicate_signal_handlers(
    qapp, wait_for_signal
):
    viewer, service = _open(qapp, wait_for_signal)
    for _ in range(3):
        viewer._renderer_combo.setCurrentText("Matplotlib")
        viewer._renderer_combo.setCurrentText("PyQtGraph")
    before = service.select_calls

    viewer.renderer.coordinate_clicked.emit(1.0)

    assert service.select_calls == before + 1  # exactly one handler connected
    assert len(viewer._splitter.widget(0).children()) > 0
    assert viewer._splitter.count() == 2  # old section widgets were removed
    assert viewer._analysis_slot.count() == 1


def test_closing_the_viewer_disposes_the_active_renderer(qapp, wait_for_signal):
    from PyQt5.QtGui import QCloseEvent

    viewer, _ = _open(qapp, wait_for_signal)
    viewer._renderer_combo.setCurrentText("Matplotlib")
    event = QCloseEvent()

    viewer.closeEvent(event)

    assert event.isAccepted() is True
    assert viewer._renderer is None


def test_pyqtgraph_renderer_uses_a_white_background_like_matplotlib(qapp):
    from PyQt5.QtGui import QColor

    renderer = PyQtGraphSeismicRenderer()
    renderer.show_section(_section(), SETTINGS)

    widgets = [renderer._layout_widget, renderer._trace_plot, renderer._spectrum_plot]
    for widget in widgets:
        assert widget.backgroundBrush().color() == QColor("white")
    renderer.dispose()
