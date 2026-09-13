"""Renderer abstraction: Matplotlib and PyQtGraph section renderers receive
the very same SeismicSection and region bounds, agree on what they draw
(shape, extent, coordinates, levels, region), and switching between them
in the viewer is presentation-only -- no service call, no worker, no I/O,
viewer-owned state (section, region, regional spectrum) kept.
"""

import inspect
import threading
from dataclasses import replace

import numpy as np
import pytest

from giecar_seismic.application.seismic_viewer import (
    SectionRegion,
    SeismicSection,
    SeismicViewerService,
    ViewerContext,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation
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


def test_each_renderer_emits_the_clicked_coordinate_only(renderer):
    section = _section()
    renderer.show_section(section, SETTINGS)
    clicked: list[float] = []
    renderer.coordinate_clicked.connect(clicked.append)

    if isinstance(renderer, MatplotlibSeismicRenderer):

        class Event:
            xdata = 2.3

        renderer._on_click(Event())
        # a toolbar zoom/pan action in progress must never select a region
        renderer._toolbar.mode = "zoom rect"
        renderer._on_click(Event())
        renderer._toolbar.mode = ""
    else:
        renderer.click_at_coordinate(2.3)

    assert clicked == [2.3]  # resolving to an axis position is the viewer's job
    assert not hasattr(renderer, "compare_coordinate_clicked")


def test_section_renderers_have_no_trace_or_spectrum_presentation(renderer):
    for name in ("show_trace", "analysis_widget", "show_compare_markers"):
        assert not hasattr(renderer, name)


# --- region highlight: same normalized bounds in both renderers ------------------


def _region_artists(renderer):
    """(spans, bound lines, start lines) drawn over every section panel."""
    if isinstance(renderer, MatplotlibSeismicRenderer):
        artists = [
            a for ax in renderer.section_figure.axes for a in (*ax.patches, *ax.lines)
        ]
        return (
            [a for a in artists if a.get_gid() == "region-span"],
            [a for a in artists if a.get_gid() == "region-bound"],
            [a for a in artists if a.get_gid() == "region-start"],
        )
    import pyqtgraph as pg

    items = [item for plot in renderer.plots for item in plot.items]
    lines = [i for i in items if isinstance(i, pg.InfiniteLine)]
    return (
        [i for i in items if isinstance(i, pg.LinearRegionItem)],
        [i for i in lines if getattr(i, "role", None) == "region-bound"],
        [i for i in lines if getattr(i, "role", None) == "region-start"],
    )


@pytest.mark.parametrize("mode", ["Original", "Side-by-side"])
def test_region_is_drawn_as_a_range_on_every_panel_and_cleared(renderer, mode):
    section = _section()
    renderer.show_section(section, replace(SETTINGS, mode=mode))
    n_panels = 2 if mode == "Side-by-side" else 1
    assert renderer.last_region is None and renderer.last_region_start is None

    renderer.show_region_start(2.0)
    assert renderer.last_region_start == 2.0 and renderer.last_region is None
    spans, bounds, starts = _region_artists(renderer)
    assert (len(spans), len(bounds), len(starts)) == (0, 0, n_panels)

    renderer.show_region(2.0, 4.0)
    assert renderer.last_region == (2.0, 4.0) and renderer.last_region_start is None
    spans, bounds, starts = _region_artists(renderer)
    assert (len(spans), len(bounds), len(starts)) == (n_panels, 2 * n_panels, 0)
    if isinstance(renderer, MatplotlibSeismicRenderer):
        xs = sorted({float(line.get_xdata()[0]) for line in bounds})
        span_bounds = {
            (round(p.get_x(), 6), round(p.get_x() + p.get_width(), 6)) for p in spans
        }
        assert span_bounds == {(2.0, 4.0)}
    else:
        xs = sorted({float(line.value()) for line in bounds})
        assert {tuple(s.getRegion()) for s in spans} == {(2.0, 4.0)}
    assert xs == [2.0, 4.0]
    # one range, not one marker per trace: region 1..4 still draws 2 bounds
    renderer.show_region(1.0, 4.0)
    _, bounds, _ = _region_artists(renderer)
    assert len(bounds) == 2 * n_panels

    renderer.clear_region()
    assert renderer.last_region is None and renderer.last_region_start is None
    assert _region_artists(renderer) == ([], [], [])

    renderer.show_region(1.0, 3.0)
    renderer.show_section(section, replace(SETTINGS, mode=mode))  # navigation
    assert renderer.last_region is None  # the viewer re-applies its own state
    assert _region_artists(renderer) == ([], [], [])
    renderer.show_section(None, SETTINGS)
    assert _region_artists(renderer) == ([], [], [])


def test_both_renderers_record_identical_region_bounds(qapp):
    mpl, pg = MatplotlibSeismicRenderer(), PyQtGraphSeismicRenderer()
    try:
        for r in (mpl, pg):
            r.show_section(_section(), SETTINGS)
            r.show_region(3.0, 1.0)  # renderers draw what they are told, normalized
        assert mpl.last_region == pg.last_region == (1.0, 3.0)
        for r in (mpl, pg):
            r.show_region_start(2.0)
        assert mpl.last_region_start == pg.last_region_start == 2.0
    finally:
        mpl.dispose()
        pg.dispose()


# --- viewer integration: switching renderer ----------------------------------------


class FakeViewerService:
    def __init__(self):
        self.load_calls: list[tuple[LineOrientation, int]] = []
        self.regional_calls = 0
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

    def regional_spectrum(self, section, region, dataset, job, **kwargs):
        self.regional_calls += 1
        return SeismicViewerService.regional_spectrum(
            section, region, dataset, job, **kwargs
        )


def _open(qapp, wait_for_signal):
    service = FakeViewerService()
    viewer = SeismicViewer(service, lambda d: True, target=7)
    for _ in range(2):
        thread = viewer._thread
        assert thread is not None
        wait_for_signal(thread.finished)
    return viewer, service


def _select_region(viewer, wait_for_signal, first: float, second: float) -> None:
    for coordinate in (first, second):
        viewer.renderer.coordinate_clicked.emit(coordinate)
        while viewer._thread is not None:  # regional spectrum worker(s)
            wait_for_signal(viewer._thread.finished)


def test_viewer_defaults_to_pyqtgraph_and_offers_matplotlib(qapp, wait_for_signal):
    viewer, _ = _open(qapp, wait_for_signal)
    assert viewer._renderer_combo.currentText() == "PyQtGraph"
    assert [viewer._renderer_combo.itemText(i) for i in range(2)] == RENDERERS
    assert isinstance(viewer.renderer, PyQtGraphSeismicRenderer)
    viewer.close()


def test_switching_renderer_keeps_state_and_does_no_io_or_worker(
    qapp, wait_for_signal, monkeypatch
):
    viewer, service = _open(qapp, wait_for_signal)
    viewer._line_spinbox.setValue(11)
    wait_for_signal(viewer._thread.finished)
    viewer._mode_combo.setCurrentText("Difference")
    viewer._gain_spinbox.setValue(1.5)
    viewer._clip_spinbox.setValue(98.0)
    viewer._cmap_combo.setCurrentText("gray")
    _select_region(viewer, wait_for_signal, 1.2, 4.0)  # inline 11: (11, 3) missing
    section_before = viewer._section
    region_before = viewer._region
    regional_before = viewer._regional
    assert regional_before is not None
    assert region_before.region == SectionRegion(1, 4)
    loads, regionals = len(service.load_calls), service.regional_calls
    old_renderer, old_spectrum_renderer = viewer.renderer, viewer.spectrum_renderer

    def forbidden(*args, **kwargs):
        pytest.fail("renderer switch performed science or I/O")

    monkeypatch.setattr(np.fft, "rfft", forbidden)
    monkeypatch.setattr(viewer, "_start_worker", forbidden)
    viewer._renderer_combo.setCurrentText("Matplotlib")

    assert isinstance(viewer.renderer, MatplotlibSeismicRenderer)
    assert viewer.renderer is not old_renderer
    assert viewer.spectrum_renderer is not old_spectrum_renderer
    assert viewer._thread is None  # no worker started
    assert len(service.load_calls) == loads and service.regional_calls == regionals
    assert viewer._section is section_before  # same object, not a copy
    assert viewer.orientation is LineOrientation.INLINE
    assert viewer._line_spinbox.value() == 11
    assert viewer._mode_combo.currentText() == "Difference"
    assert viewer._gain_spinbox.value() == 1.5
    assert viewer._clip_spinbox.value() == 98.0
    assert viewer._cmap_combo.currentText() == "gray"
    assert viewer._region is region_before and viewer._regional is regional_before
    assert viewer._renderer_combo.isEnabled() is True
    # the new renderer drew the same section, region and regional spectrum
    panel = viewer.renderer.last_panels[0]
    assert panel.title.startswith("Filtered - Original -- inline 11")
    assert panel.levels == (
        -amplitude_limit(section_before, 98.0, 1.5),
        amplitude_limit(section_before, 98.0, 1.5),
    )
    assert viewer.renderer.last_region == (1.0, 4.0)
    assert viewer.spectrum_renderer.last_spectrum is viewer._display.spectrum
    assert "XL 1–4" in viewer._region_label.text()
    viewer.close()


def test_region_resolves_identically_in_both_renderers(qapp, wait_for_signal):
    viewer, _ = _open(qapp, wait_for_signal)
    viewer._line_spinbox.setValue(11)
    wait_for_signal(viewer._thread.finished)

    results = {}
    for name in RENDERERS:
        viewer._renderer_combo.setCurrentText(name)
        _select_region(viewer, wait_for_signal, 3.0, 1.2)  # 3 is the gap, still a bound
        assert viewer._regional is not None
        results[name] = (
            viewer._region.region,
            viewer._region.trace_indices,
            viewer.renderer.last_region,
        )

    assert results["Matplotlib"] == results["PyQtGraph"]
    assert results["PyQtGraph"] == (SectionRegion(1, 3), (13, 14), (1.0, 3.0))
    viewer.close()


def test_switching_back_and_forth_does_not_duplicate_signal_handlers(
    qapp, wait_for_signal
):
    viewer, _ = _open(qapp, wait_for_signal)
    for _ in range(3):
        viewer._renderer_combo.setCurrentText("Matplotlib")
        viewer._renderer_combo.setCurrentText("PyQtGraph")

    viewer.renderer.coordinate_clicked.emit(1.0)

    assert viewer._region_start == 1  # exactly one handler: still ONE boundary
    assert viewer._region.region == SectionRegion(1, 1)
    while viewer._thread is not None:
        wait_for_signal(viewer._thread.finished)
    assert len(viewer._splitter.widget(0).children()) > 0
    assert viewer._splitter.count() == 2  # old section widgets were removed
    assert viewer._spectrum_slot.count() == 1
    viewer.close()


def test_closing_the_viewer_disposes_the_active_renderers(qapp, wait_for_signal):
    from PyQt5.QtGui import QCloseEvent

    viewer, _ = _open(qapp, wait_for_signal)
    viewer._renderer_combo.setCurrentText("Matplotlib")
    event = QCloseEvent()

    viewer.closeEvent(event)

    assert event.isAccepted() is True
    assert viewer._renderer is None
    assert viewer._spectrum_renderer is None


def test_pyqtgraph_renderer_uses_a_white_background_like_matplotlib(qapp):
    from PyQt5.QtGui import QColor

    renderer = PyQtGraphSeismicRenderer()
    renderer.show_section(_section(), SETTINGS)

    assert renderer._layout_widget.backgroundBrush().color() == QColor("white")
    renderer.dispose()
