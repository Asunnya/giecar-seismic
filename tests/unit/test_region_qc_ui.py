"""Region-oriented spectral QC in the viewer: one normal click analyses a
single trace (a region of width one), a second click extends it to the
contiguous region between the clicks; every physically present trace
inside feeds ONE aggregate original vs ONE aggregate filtered spectrum,
computed by a QThread worker from the section already in memory (no I/O)
and shown directly in the right-hand panel.

Fixture (test_seismic_renderers): inline 10 has crosslines 1..4 (physical
0..3); inline 11 has (11, 3) missing (physical 13, 14, -1, 16).
"""

import inspect
import threading

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from giecar_seismic.application.seismic_viewer import (
    RegionalSpectrum,
    SectionRegion,
    SeismicViewerService,
    SpectrumScale,
    regional_spectrum_for_display,
)
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.ui import region_spectrum_window as window_module
from giecar_seismic.ui import seismic_viewer as viewer_module
from giecar_seismic.ui.region_spectrum_window import RegionSpectrumWindow
from giecar_seismic.ui.seismic_viewer import REGION_PROMPT, SeismicViewer
from giecar_seismic.ui.viewer_workers import RegionalSpectrumWorker
from tests.unit.test_seismic_renderers import (
    FakeViewerService,
    _dataset,
    _job,
    _region_artists,
    _section,
)


class BlockingRegionalService(FakeViewerService):
    """Regional aggregation that can be held open, and records the thread
    it ran on, so tests can prove it never runs on the GUI thread."""

    def __init__(self):
        super().__init__()
        self.regional_entered = threading.Event()
        self.regional_release = threading.Event()
        self.regional_release.set()
        self.regional_threads: list[str] = []
        self.regional_regions: list[SectionRegion] = []
        self.fail_with: str | None = None

    def regional_spectrum(self, section, region, dataset, job, **kwargs):
        self.regional_threads.append(threading.current_thread().name)
        self.regional_regions.append(region)
        self.regional_entered.set()
        assert self.regional_release.wait(timeout=5), "test deadlocked"
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        return super().regional_spectrum(section, region, dataset, job, **kwargs)


def _open(qapp, wait_for_signal, service=None):
    service = service or BlockingRegionalService()
    viewer = SeismicViewer(service, lambda d: True, target=7)
    for _ in range(2):
        thread = viewer._thread
        assert thread is not None
        wait_for_signal(thread.finished)
    return viewer, service


@pytest.fixture
def opened(qapp, wait_for_signal):
    viewer, service = _open(qapp, wait_for_signal)
    yield viewer, service
    service.regional_release.set()
    service.release.set()
    if viewer._thread is not None:
        wait_for_signal(viewer._thread.finished)
    viewer.close()


def click(viewer, coordinate: float) -> None:
    """Through the renderer's own click signal, as a real click would."""
    viewer.renderer.click_at_coordinate(coordinate)


def _settle(viewer, wait_for_signal) -> None:
    while viewer._thread is not None:
        wait_for_signal(viewer._thread.finished)


def select_region(viewer, wait_for_signal, first: float, second: float) -> None:
    click(viewer, first)
    _settle(viewer, wait_for_signal)
    click(viewer, second)
    _settle(viewer, wait_for_signal)


def load_line(viewer, wait_for_signal, line: int) -> None:
    viewer._line_spinbox.setValue(line)
    wait_for_signal(viewer._thread.finished)


# --- interaction: two clicks, one region ----------------------------------------


def test_initial_panel_asks_for_two_region_boundaries_not_a_trace(opened):
    viewer, _ = opened
    text = viewer._region_label.text()
    assert text == REGION_PROMPT
    assert "Click a trace" in text and "second position" in text
    assert viewer._region_start is None and viewer.region is None
    assert viewer.regional_spectrum is None
    assert viewer.spectrum_renderer.last_spectrum is None
    assert not viewer._open_spectrum_button.isEnabled()


def test_one_click_analyses_that_single_trace(opened, wait_for_signal):
    viewer, service = opened

    click(viewer, 2.3)  # snaps to crossline 2 (physical 1)
    assert viewer._region_start == 2
    assert viewer.region is not None and viewer.region.region == SectionRegion(2, 2)
    assert isinstance(viewer._worker, RegionalSpectrumWorker)
    assert viewer.renderer.last_region == (2.0, 2.0)
    _settle(viewer, wait_for_signal)

    regional = viewer.regional_spectrum
    assert regional is not None and service.regional_calls == 1
    assert regional.region.is_single_trace
    assert regional.region.trace_indices == (1,)
    text = viewer._region_label.text()
    assert "Region: XL 2–2" in text and "1 trace present" in text
    assert "Single trace (physical index 1)" in text
    assert "Click a second boundary to extend the region." in text
    np.testing.assert_array_equal(
        viewer.spectrum_renderer.last_spectrum.original,
        np.abs(np.fft.rfft(viewer._section.original[1])),
    )
    assert viewer._open_spectrum_button.isEnabled()


def test_second_click_completes_the_region_and_starts_one_worker_off_the_gui_thread(
    opened, wait_for_signal
):
    viewer, service = opened
    click(viewer, 1.0)
    _settle(viewer, wait_for_signal)
    service.regional_release.clear()
    service.regional_entered.clear()

    click(viewer, 3.0)

    assert viewer._region_start is None
    assert viewer.region is not None
    assert viewer.region.region == SectionRegion(1, 3)
    thread = viewer._thread
    assert thread is not None  # exactly one worker in flight
    assert isinstance(viewer._worker, RegionalSpectrumWorker)
    assert "Calculating regional spectrum" in viewer._region_label.text()
    assert viewer.renderer.last_region == (1.0, 3.0)
    assert service.regional_entered.wait(timeout=5)
    # one call per click (single trace, then the region), each on a worker
    # QThread -- never the GUI thread. Thread names are not compared across
    # calls: every worker gets its own QThread and the OS may or may not
    # reuse the underlying thread (it does on Linux, not on Windows).
    assert len(service.regional_threads) == 2
    assert threading.current_thread().name not in service.regional_threads
    # responsive meanwhile: widgets answer; a further click is queued, not
    # run concurrently and not lost
    click(viewer, 4.0)
    assert viewer._region_start is None and viewer.region.region == SectionRegion(1, 3)
    assert viewer._pending_click == 4.0
    assert viewer._line_spinbox.isEnabled() is False  # existing safe policy
    assert viewer.regional_spectrum is None  # result not applied yet

    service.regional_release.set()
    wait_for_signal(thread.finished)

    assert service.regional_regions[-1] == SectionRegion(1, 3)  # computed once
    # the queued click then started over as a single trace at 4
    assert viewer._region_start == 4 and viewer._thread is not None
    assert viewer.region.region == SectionRegion(4, 4)
    _settle(viewer, wait_for_signal)
    assert viewer.regional_spectrum.region.trace_indices == (3,)
    assert viewer._thread is None and viewer._worker is None
    assert viewer._line_spinbox.isEnabled() is True


@pytest.mark.parametrize("order", [(1.0, 4.0), (4.0, 1.0)])
def test_click_order_does_not_matter(opened, wait_for_signal, order):
    viewer, service = opened
    select_region(viewer, wait_for_signal, *order)
    assert viewer.region.region == SectionRegion(1, 4)
    first = int(order[0])
    assert service.regional_regions == [
        SectionRegion(first, first),
        SectionRegion(1, 4),
    ]
    assert viewer.renderer.last_region == (1.0, 4.0)


def test_result_panel_shows_bounds_counts_and_both_aggregate_curves(
    opened, wait_for_signal
):
    viewer, _ = opened
    load_line(viewer, wait_for_signal, 11)  # (11, 3) missing

    select_region(viewer, wait_for_signal, 4.0, 1.0)

    regional = viewer.regional_spectrum
    assert isinstance(regional, RegionalSpectrum)
    assert regional.region.trace_indices == (13, 14, 16)
    assert (regional.n_positions, regional.n_present, regional.n_missing) == (4, 3, 1)
    text = viewer._region_label.text()
    assert "Inline 11" in text
    assert "Region: XL 1–4" in text
    assert "3 traces present" in text
    assert "1 missing position" in text
    assert "Calculating" not in text
    expected = SeismicViewerService.regional_spectrum(
        viewer._section, SectionRegion(1, 4), _dataset(), _job()
    )
    np.testing.assert_array_equal(
        regional.spectrum.original, expected.spectrum.original
    )
    np.testing.assert_array_equal(
        regional.spectrum.filtered, expected.spectrum.filtered
    )
    drawn = viewer.spectrum_renderer.last_spectrum
    assert drawn is viewer._display.spectrum
    np.testing.assert_array_equal(drawn.original, regional.spectrum.original)
    np.testing.assert_array_equal(drawn.filtered, regional.spectrum.filtered)
    assert [m.frequency_hz for m in drawn.cutoff_markers] == [30.0]
    assert viewer.spectrum_renderer.visibility.original
    assert viewer.spectrum_renderer.visibility.filtered
    assert viewer._open_spectrum_button.isEnabled()


def test_third_click_starts_a_fresh_region_and_clears_the_previous_aggregate(
    opened, wait_for_signal
):
    viewer, service = opened
    select_region(viewer, wait_for_signal, 1.0, 2.0)
    assert viewer.regional_spectrum is not None

    previous = viewer.regional_spectrum
    click(viewer, 4.0)

    assert viewer._region_start == 4
    assert viewer.region.region == SectionRegion(4, 4)  # fresh single trace
    assert viewer.regional_spectrum is None  # previous aggregate cleared
    assert viewer.spectrum_renderer.last_spectrum is None
    assert viewer.renderer.last_region == (4.0, 4.0)
    _settle(viewer, wait_for_signal)
    assert viewer.regional_spectrum is not previous
    assert viewer.regional_spectrum.region.trace_indices == (3,)

    click(viewer, 3.0)
    _settle(viewer, wait_for_signal)
    assert viewer.region.region == SectionRegion(3, 4)
    assert viewer.regional_spectrum.region.trace_indices == (2, 3)
    assert service.regional_calls == 4


@pytest.mark.parametrize("navigate", ["next_inline", "spinbox", "orientation"])
def test_navigating_clears_region_and_spectrum(opened, wait_for_signal, navigate):
    viewer, _ = opened
    select_region(viewer, wait_for_signal, 1.0, 3.0)
    viewer._open_spectrum_button.click()
    window = viewer._region_window
    assert window is not None and window.regional is not None

    if navigate == "next_inline":
        viewer._next_button.click()
    elif navigate == "spinbox":
        viewer._line_spinbox.setValue(12)
    else:
        viewer._orientation_combo.setCurrentIndex(1)  # crossline
    assert viewer.region is None and viewer.regional_spectrum is None  # at request
    wait_for_signal(viewer._thread.finished)

    assert viewer._region_start is None and viewer.region is None
    assert viewer.regional_spectrum is None
    assert viewer._region_label.text() == REGION_PROMPT
    assert viewer.renderer.last_region is None
    assert _region_artists(viewer.renderer) == ([], [], [])
    assert viewer.spectrum_renderer.last_spectrum is None
    assert window.regional is None and window.renderer.last_spectrum is None
    if navigate == "orientation":
        assert viewer._section.orientation is LineOrientation.CROSSLINE
        click(viewer, 10.0)
        assert "Region: IL 10–10" in viewer._region_label.text()
        _settle(viewer, wait_for_signal)


def test_a_single_present_trace_next_to_a_gap_is_still_a_single_trace_region(
    opened, wait_for_signal
):
    viewer, _ = opened
    load_line(viewer, wait_for_signal, 11)  # 1 -> 13, 2 -> 14, 3 -> gap, 4 -> 16

    select_region(viewer, wait_for_signal, 3.0, 4.0)

    regional = viewer.regional_spectrum
    assert regional is not None and regional.region.trace_indices == (16,)
    assert regional.region.is_single_trace
    text = viewer._region_label.text()
    assert "Region: XL 3–4" in text
    assert "1 trace present" in text and "1 missing position" in text
    assert "Single trace (physical index 16)" in text
    assert "extend" not in text  # region complete: no pending boundary


def test_region_without_any_trace_gives_feedback_and_no_worker(opened, wait_for_signal):
    viewer, service = opened
    load_line(viewer, wait_for_signal, 11)  # 3 is the gap

    click(viewer, 3.0)  # gap only: nothing to analyse yet

    assert viewer._thread is None and service.regional_calls == 0
    assert viewer._region_start == 3  # a second click may still extend it
    assert viewer.region is not None and not viewer.region.is_valid
    assert viewer.regional_spectrum is None
    text = viewer._region_label.text()
    assert "Region: XL 3–3" in text
    assert "0 traces present" in text and "1 missing position" in text
    assert "Region must contain at least 1 trace." in text
    assert viewer.renderer.last_region == (3.0, 3.0)  # still indicated

    click(viewer, 4.0)  # extend over the gap: one trace -> valid
    _settle(viewer, wait_for_signal)
    assert viewer.regional_spectrum.region.trace_indices == (16,)


def test_gaps_inside_a_region_do_not_prevent_the_calculation(opened, wait_for_signal):
    viewer, _ = opened
    load_line(viewer, wait_for_signal, 11)
    select_region(viewer, wait_for_signal, 2.0, 4.0)  # 14, gap, 16
    regional = viewer.regional_spectrum
    assert regional is not None
    assert regional.region.trace_indices == (14, 16)
    assert (regional.n_present, regional.n_missing) == (2, 1)
    assert np.isfinite(regional.spectrum.original).all()


def test_clicks_outside_the_axis_are_ignored(opened):
    viewer, _ = opened
    click(viewer, 42.0)
    assert viewer._region_start is None
    assert viewer._region_label.text() == REGION_PROMPT


# --- no I/O, no FFT on presentation changes -----------------------------------------


def test_region_selection_and_result_cause_no_io_or_section_load(opened, monkeypatch):
    viewer, service = opened

    def forbidden(*args, **kwargs):
        pytest.fail("region QC touched acquisition or the section loader")

    for name in ("context", "load_section", "line_numbers"):
        monkeypatch.setattr(service, name, forbidden)
    loads = len(service.load_calls)

    click(viewer, 1.0)
    assert isinstance(viewer._worker, RegionalSpectrumWorker)
    click(viewer, 4.0)
    assert viewer._pending_click == 4.0

    assert len(service.load_calls) == loads


def test_presentation_toggles_use_the_cached_regional_spectrum(
    opened, wait_for_signal, monkeypatch
):
    viewer, service = opened
    select_region(viewer, wait_for_signal, 1.0, 3.0)
    raw = viewer.regional_spectrum
    assert raw is not None

    def forbidden(*args, **kwargs):
        pytest.fail("presentation change performed science, I/O or a worker")

    monkeypatch.setattr(np.fft, "rfft", forbidden)
    monkeypatch.setattr(viewer, "_start_worker", forbidden)
    for name in ("context", "load_section", "regional_spectrum"):
        monkeypatch.setattr(service, name, forbidden)

    viewer._spectrum_scale_combo.setCurrentText("dB")
    shown = viewer.spectrum_renderer.last_spectrum
    assert shown.scale is SpectrumScale.DB
    assert shown.original[np.argmax(raw.spectrum.original)] == 0
    viewer._response_checkbox.setChecked(True)
    assert viewer.spectrum_renderer.last_spectrum.show_filter_response
    assert viewer.spectrum_renderer.visibility.response
    viewer._original_checkbox.setChecked(False)
    assert not viewer.spectrum_renderer.visibility.original
    viewer._renderer_combo.setCurrentText("Matplotlib")
    assert viewer.spectrum_renderer.last_spectrum is viewer._display.spectrum
    assert viewer.renderer.last_region == (1.0, 3.0)
    viewer._spectrum_scale_combo.setCurrentText("Linear")
    assert viewer.regional_spectrum is raw  # cached, never rebuilt
    assert viewer._display.spectrum.original is raw.spectrum.original
    expected = regional_spectrum_for_display(raw, SpectrumScale.LINEAR)
    np.testing.assert_array_equal(
        viewer.spectrum_renderer.last_spectrum.filtered, expected.spectrum.filtered
    )


# --- detached regional window ----------------------------------------------------


def test_open_spectrum_shows_the_current_region_without_io_or_fft(
    opened, wait_for_signal, monkeypatch
):
    viewer, service = opened
    assert not viewer._open_spectrum_button.isEnabled()
    viewer._open_spectrum_button.click()
    assert viewer._region_window is None
    select_region(viewer, wait_for_signal, 1.0, 4.0)
    raw = viewer.regional_spectrum

    def forbidden(*args, **kwargs):
        pytest.fail("detached window touched acquisition, science or a worker")

    monkeypatch.setattr(np.fft, "rfft", forbidden)
    monkeypatch.setattr(viewer, "_start_worker", forbidden)
    for name in ("context", "load_section", "regional_spectrum"):
        monkeypatch.setattr(service, name, forbidden)
    viewer._open_spectrum_button.click()
    window = viewer._region_window

    assert isinstance(window, RegionSpectrumWindow)
    assert window.isVisible() and not window.isModal()
    assert window.windowModality() == Qt.NonModal
    assert window.regional is raw
    assert "Region: XL 1–4" in window._identity_label.text()
    assert "4 traces present" in window._identity_label.text()
    assert RegionalSpectrum.METHOD in window._method_label.text()
    assert "Compare" not in window.windowTitle()
    for fragment in ("Low-pass", "30", "Order: 4", "4.0 ms", "125"):
        assert fragment in window._metadata_label.text()
    window._scale_combo.setCurrentText("dB")
    assert window.display.spectrum.scale is SpectrumScale.DB
    window._renderer_combo.setCurrentText("Matplotlib")
    assert window.renderer.last_spectrum is window.display.spectrum
    viewer._open_spectrum_button.click()
    assert viewer._region_window is window  # reused

    window.close()
    assert viewer._region_window is None
    assert viewer.regional_spectrum is raw  # closing the window keeps the QC


def test_closing_the_viewer_closes_the_region_window(qapp, wait_for_signal):
    viewer, _ = _open(qapp, wait_for_signal)
    select_region(viewer, wait_for_signal, 1.0, 2.0)
    viewer._open_spectrum_button.click()
    window = viewer._region_window
    assert window is not None
    viewer.close()
    assert not window.isVisible()
    assert viewer._region_window is None


# --- worker lifecycle ------------------------------------------------------------


def test_worker_failure_becomes_a_signal_and_the_panel_recovers(
    opened, wait_for_signal
):
    viewer, service = opened
    service.fail_with = "boom"

    select_region(viewer, wait_for_signal, 1.0, 3.0)

    assert viewer._thread is None and viewer._worker is None
    assert viewer.regional_spectrum is None
    assert "Regional spectrum failed: boom" in viewer._region_label.text()
    assert viewer.region is not None  # region stays indicated
    assert viewer._line_spinbox.isEnabled()
    service.fail_with = None
    click(viewer, 2.0)  # region interaction restored
    assert viewer._region_start == 2
    _settle(viewer, wait_for_signal)
    assert viewer.regional_spectrum is not None
    click(viewer, 4.0)
    _settle(viewer, wait_for_signal)
    assert viewer.regional_spectrum.region.region == SectionRegion(2, 4)


def test_close_is_rejected_while_the_regional_worker_runs(opened, wait_for_signal):
    from PyQt5.QtGui import QCloseEvent

    viewer, service = opened
    service.regional_release.clear()
    service.regional_entered.clear()
    click(viewer, 1.0)
    assert service.regional_entered.wait(timeout=5)

    event = QCloseEvent()
    viewer.closeEvent(event)
    assert event.isAccepted() is False
    assert "Waiting for the current load" in viewer._status_label.text()

    thread = viewer._thread
    service.regional_release.set()
    wait_for_signal(thread.finished)
    idle = QCloseEvent()
    viewer.closeEvent(idle)
    assert idle.isAccepted() is True


def test_regional_worker_alone_emits_computed_or_failed(qapp):
    section = _section()
    computed: list[RegionalSpectrum] = []
    failed: list[str] = []
    ok = RegionalSpectrumWorker(
        SeismicViewerService, section, SectionRegion(1, 3), _dataset(), _job()
    )
    ok.computed.connect(computed.append)
    ok.run()
    assert computed[0].region.trace_indices == (0, 1, 2)

    bad = RegionalSpectrumWorker(
        SeismicViewerService, section, SectionRegion(9, 9), _dataset(), _job()
    )
    bad.failed.connect(failed.append)
    bad.run()  # off-axis region: ValueError becomes a signal, never escapes
    assert failed and "axis" in failed[0]


# --- the old trace-oriented surface is gone ---------------------------------------


def test_no_trace_or_comparison_workflow_remains():
    source = inspect.getsource(viewer_module)
    for token in (
        "Click a trace on the section.",
        "Compare Spectra",
        "MAX_COMPARE_TRACES",
        "compare_coordinate_clicked",
        "_compare_indices",
        "TraceView",
        "select_trace",
        "_trace_info_label",
        "show_trace",
    ):
        assert token not in source, token
    for name in ("toggle_compare_coordinate", "compare_indices", "_selected"):
        assert not hasattr(SeismicViewer, name)
    assert "Compare" not in inspect.getsource(window_module)


def test_viewer_and_window_modules_know_no_infrastructure_or_science():
    for module in (viewer_module, window_module):
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
            or "scipy" in l
            for l in lines
        )
        assert "rfft" not in inspect.getsource(module)
    assert "numpy" not in " ".join(
        l
        for l in inspect.getsource(window_module).splitlines()
        if l.startswith(("import ", "from "))
    )


# --- single-trace waveform (amplitude x time) in the panel --------------------------


def test_single_trace_shows_its_waveform_and_a_region_hides_it(opened, wait_for_signal):
    viewer, _ = opened
    click(viewer, 2.0)
    _settle(viewer, wait_for_signal)

    waveform = viewer.regional_spectrum.waveform
    assert waveform is not None and waveform.trace_index == 1
    assert viewer.spectrum_renderer.last_waveform is waveform
    assert not viewer.spectrum_renderer.waveform_widget().isHidden()
    np.testing.assert_array_equal(waveform.original, viewer._section.original[1])
    viewer._open_spectrum_button.click()
    assert viewer._region_window.renderer.last_waveform is waveform

    click(viewer, 4.0)  # extend: region of 3 traces -> no time-domain curve
    _settle(viewer, wait_for_signal)
    assert viewer.regional_spectrum.waveform is None
    assert viewer.spectrum_renderer.last_waveform is None
    assert viewer.spectrum_renderer.waveform_widget().isHidden()
    assert viewer._region_window.renderer.last_waveform is None

    viewer._renderer_combo.setCurrentText("Matplotlib")
    click(viewer, 3.0)
    _settle(viewer, wait_for_signal)
    assert viewer.spectrum_renderer.last_waveform.trace_index == 2
    viewer._renderer_combo.setCurrentText("PyQtGraph")  # presentation only
    assert viewer.spectrum_renderer.last_waveform.trace_index == 2
