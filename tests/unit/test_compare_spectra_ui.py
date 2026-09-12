"""Multi-trace spectral QC in the viewer: a Ctrl+click comparison set kept
apart from the active trace, compact controls, section markers, and a
modeless window showing ONE aggregate original vs ONE aggregate filtered
spectrum computed from the loaded section only (no I/O, no worker).

Fixture: inline 10 has crosslines 1..4 (physical 0..3); inline 11 has
(11, 3) missing (physical 13, 14, -1, 16). See test_seismic_renderers.
"""

import inspect

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from giecar_seismic.application.seismic_viewer import (
    AggregateSpectrum,
    SeismicSection,
    SeismicViewerService,
    SpectrumScale,
    aggregate_for_display,
)
from giecar_seismic.ui import compare_spectrum_window as window_module
from giecar_seismic.ui.compare_spectrum_window import CompareSpectrumWindow
from giecar_seismic.ui.seismic_viewer import MAX_COMPARE_TRACES, SeismicViewer
from tests.unit.test_seismic_renderers import (
    N_SAMPLES,
    FakeViewerService,
    _dataset,
    _job,
    _marker_lines,
)


class CompareFakeService(FakeViewerService):
    """The renderer-test fake plus the real, pure aggregation."""

    aggregate_spectrum = staticmethod(SeismicViewerService.aggregate_spectrum)


class WideFakeService(CompareFakeService):
    """Inline 10 with MAX_COMPARE_TRACES + 2 traces, to exercise the limit."""

    def load_section(self, target, orientation, line_number):
        self.load_calls.append((orientation, line_number))
        n = MAX_COMPARE_TRACES + 2
        rng = np.random.default_rng(1)
        original = rng.standard_normal((n, N_SAMPLES)).astype(np.float32)
        return SeismicSection(
            orientation,
            line_number,
            np.arange(1, n + 1),
            np.arange(n),
            original,
            (original * 0.5).astype(np.float32),
            4.0,
        )


def _open(qapp, wait_for_signal, service=None):
    service = service or CompareFakeService()
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
    if viewer._thread is not None:
        wait_for_signal(viewer._thread.finished)
    viewer.close()


def ctrl_click(viewer, coordinate: float) -> None:
    """Through the renderer's own signal, as a real Ctrl+click would."""
    viewer.renderer.click_at_coordinate(coordinate, compare=True)


# --- selection state ----------------------------------------------------------


def test_normal_click_keeps_the_single_trace_workflow_untouched(opened):
    viewer, service = opened
    viewer.renderer.click_at_coordinate(2.0)

    assert viewer._selected is not None
    assert viewer._selected.geometry.trace_index == 1
    assert viewer._open_spectrum_button.isEnabled()
    assert viewer.compare_indices == ()  # a normal click never touches the set
    assert viewer._compare_button.isHidden()
    assert service.select_calls == 1


def test_ctrl_click_toggles_traces_in_and_out_by_physical_identity(opened):
    viewer, service = opened
    before = service.select_calls

    ctrl_click(viewer, 2.0)
    assert viewer.compare_indices == (1,)
    assert viewer._selected is None  # independent from the active trace
    ctrl_click(viewer, 3.4)  # snaps to crossline 3
    ctrl_click(viewer, 0.9)  # snaps to crossline 1
    assert viewer.compare_indices == (1, 2, 0)  # selection order
    ctrl_click(viewer, 3.0)  # again -> removed
    assert viewer.compare_indices == (1, 0)
    ctrl_click(viewer, 1.2)  # same trace, different float -> no duplicate
    assert viewer.compare_indices == (1,)
    assert service.select_calls == before  # no select_trace, no geometry lookup


def test_ctrl_click_on_a_gap_or_outside_the_axis_adds_nothing(opened, wait_for_signal):
    viewer, _ = opened
    viewer._line_spinbox.setValue(11)  # (11, 3) missing
    wait_for_signal(viewer._thread.finished)

    ctrl_click(viewer, 3.0)  # the gap
    ctrl_click(viewer, 9.0)  # outside the axis
    assert viewer.compare_indices == ()
    ctrl_click(viewer, 2.0)
    assert viewer.compare_indices == (14,)


def test_count_label_and_compare_button_follow_the_selection(opened):
    viewer, _ = opened
    assert viewer._compare_label.isHidden()
    assert viewer._compare_button.isHidden()

    ctrl_click(viewer, 1.0)
    assert viewer._compare_label.text() == "1 trace selected"
    assert viewer._compare_button.isHidden()  # fewer than 2

    ctrl_click(viewer, 2.0)
    assert viewer._compare_label.text() == "2 traces selected"
    assert not viewer._compare_button.isHidden() and viewer._compare_button.isEnabled()
    assert viewer._compare_button.text() == "Compare Spectra (2)"

    ctrl_click(viewer, 2.0)
    assert viewer._compare_label.text() == "1 trace selected"
    assert viewer._compare_button.isHidden()
    ctrl_click(viewer, 1.0)
    assert viewer._compare_label.isHidden()


def test_selection_is_capped_at_max_compare_traces_with_feedback(qapp, wait_for_signal):
    viewer, _ = _open(qapp, wait_for_signal, WideFakeService())
    try:
        for coordinate in range(1, MAX_COMPARE_TRACES + 1):
            ctrl_click(viewer, float(coordinate))
        assert len(viewer.compare_indices) == MAX_COMPARE_TRACES

        ctrl_click(viewer, float(MAX_COMPARE_TRACES + 1))

        assert len(viewer.compare_indices) == MAX_COMPARE_TRACES
        assert MAX_COMPARE_TRACES + 1 - 1 not in viewer.compare_indices
        assert (
            f"Limit of {MAX_COMPARE_TRACES} traces reached"
            in viewer._compare_label.text()
        )
        assert (
            viewer._compare_button.text() == f"Compare Spectra ({MAX_COMPARE_TRACES})"
        )
        ctrl_click(viewer, 1.0)  # removing still works at the limit
        assert len(viewer.compare_indices) == MAX_COMPARE_TRACES - 1
        assert "Limit" not in viewer._compare_label.text()
    finally:
        viewer.close()


@pytest.mark.parametrize("navigate", ["next", "spinbox", "orientation"])
def test_loading_another_line_or_orientation_clears_the_selection(
    opened, wait_for_signal, navigate
):
    viewer, _ = opened
    ctrl_click(viewer, 1.0)
    ctrl_click(viewer, 2.0)
    assert viewer.compare_indices == (0, 1)
    assert viewer.renderer.last_compare_markers == (1.0, 2.0)

    if navigate == "next":
        viewer._next_button.click()
    elif navigate == "spinbox":
        viewer._line_spinbox.setValue(12)
    else:
        viewer._orientation_combo.setCurrentIndex(1)
    assert viewer.compare_indices == ()  # cleared as soon as the load is requested
    wait_for_signal(viewer._thread.finished)

    assert viewer.compare_indices == ()
    assert viewer._compare_button.isHidden()
    assert viewer.renderer.last_compare_markers == ()
    assert _marker_lines(viewer.renderer) == []


def test_markers_follow_the_selection_and_survive_redraws(opened):
    viewer, _ = opened
    ctrl_click(viewer, 1.0)
    ctrl_click(viewer, 3.0)
    assert viewer.renderer.last_compare_markers == (1.0, 3.0)
    assert len(_marker_lines(viewer.renderer)) == 2

    viewer._mode_combo.setCurrentText("Side-by-side")  # section redrawn
    assert viewer.renderer.last_compare_markers == (1.0, 3.0)
    assert len(_marker_lines(viewer.renderer)) == 4  # both panels

    viewer._renderer_combo.setCurrentText("Matplotlib")  # renderer replaced
    assert viewer.renderer.last_compare_markers == (1.0, 3.0)
    assert len(_marker_lines(viewer.renderer)) == 4

    ctrl_click(viewer, 3.0)
    assert viewer.renderer.last_compare_markers == (1.0,)
    assert len(_marker_lines(viewer.renderer)) == 2


# --- comparison window --------------------------------------------------------


def _select_and_compare(viewer, coordinates=(1.0, 3.0, 4.0)) -> CompareSpectrumWindow:
    for coordinate in coordinates:
        ctrl_click(viewer, coordinate)
    viewer._compare_button.click()
    window = viewer._compare_window
    assert window is not None
    return window


def test_compare_opens_a_modeless_window_with_the_aggregate_of_the_selection(opened):
    viewer, _ = opened
    window = _select_and_compare(viewer)

    assert window.isVisible() and not window.isModal()
    assert window.windowModality() == Qt.NonModal
    aggregate = window.aggregate
    assert isinstance(aggregate, AggregateSpectrum)
    assert aggregate.trace_indices == (0, 2, 3)
    assert aggregate.coordinates == (1, 3, 4)
    assert aggregate.n_traces == 3
    expected = SeismicViewerService.aggregate_spectrum(
        viewer._section, [0, 2, 3], _dataset(), _job()
    )
    np.testing.assert_array_equal(
        aggregate.spectrum.original, expected.spectrum.original
    )
    np.testing.assert_array_equal(
        aggregate.spectrum.filtered, expected.spectrum.filtered
    )
    # the renderer draws exactly two aggregate curves -- never 2 * N
    drawn = window.renderer.last_spectrum
    assert drawn is window.display.spectrum
    np.testing.assert_array_equal(drawn.original, aggregate.spectrum.original)
    np.testing.assert_array_equal(drawn.filtered, aggregate.spectrum.filtered)
    assert [m.frequency_hz for m in drawn.cutoff_markers] == [30.0]
    assert AggregateSpectrum.METHOD in window._method_label.text()
    assert "3 traces" in window._method_label.text()
    assert "Inline 10" in window._identity_label.text()
    assert "1, 3, 4" in window._identity_label.text()
    for fragment in ("Low-pass", "30", "Order: 4", "4.0 ms", "125"):
        assert fragment in window._metadata_label.text()
    # the single-trace window is a different thing and was not opened
    assert viewer._spectrum_window is None
    # the compare button is reused: same window, refreshed aggregate
    viewer._compare_button.click()
    assert viewer._compare_window is window


def test_opening_and_toggling_the_window_does_no_io_and_no_fft(opened, monkeypatch):
    viewer, service = opened
    for coordinate in (1.0, 2.0):
        ctrl_click(viewer, coordinate)

    def forbidden(*args, **kwargs):
        pytest.fail("comparison touched acquisition, a worker or the service")

    for name in ("context", "load_section", "select_trace", "spectrum"):
        monkeypatch.setattr(service, name, forbidden)
    monkeypatch.setattr(viewer, "_start_worker", forbidden)
    viewer._compare_button.click()
    window = viewer._compare_window
    assert window is not None and window.aggregate is not None
    raw = window.aggregate

    monkeypatch.setattr(np.fft, "rfft", forbidden)  # presentation only from here
    window._scale_combo.setCurrentText("dB")
    assert window.display.spectrum.scale is SpectrumScale.DB
    assert window.display.spectrum.original[np.argmax(raw.spectrum.original)] == 0
    window._response_checkbox.setChecked(True)
    assert window.renderer.last_spectrum.show_filter_response
    window._original_checkbox.setChecked(False)
    assert not window.renderer.visibility.original
    window._renderer_combo.setCurrentText("Matplotlib")
    assert window.renderer.last_spectrum is window.display.spectrum
    window._scale_combo.setCurrentText("Linear")
    assert window.aggregate is raw  # the raw aggregate is cached, never rebuilt
    assert window.display.spectrum.original is raw.spectrum.original


@pytest.mark.parametrize("scale", list(SpectrumScale))
def test_both_spectrum_renderers_receive_the_same_aggregate_arrays(opened, scale):
    viewer, _ = opened
    window = _select_and_compare(viewer, (1.0, 2.0))
    window._scale_combo.setCurrentText(scale.value)
    window._response_checkbox.setChecked(True)
    expected = aggregate_for_display(window.aggregate, scale, show_filter_response=True)

    seen = {}
    for name in ("PyQtGraph", "Matplotlib"):
        window._renderer_combo.setCurrentText(name)
        renderer = window.renderer
        np.testing.assert_array_equal(
            renderer.last_spectrum.original, expected.spectrum.original
        )
        np.testing.assert_array_equal(
            renderer.last_spectrum.filtered, expected.spectrum.filtered
        )
        np.testing.assert_array_equal(
            renderer.last_spectrum.filter_response, expected.spectrum.filter_response
        )
        if name == "Matplotlib":
            ax = renderer.figure.axes[0]
            np.testing.assert_array_equal(
                ax.lines[0].get_ydata(), expected.spectrum.original
            )
            seen[name] = [line.get_xdata()[0] for line in ax.lines[2:]]
        else:
            np.testing.assert_array_equal(
                renderer.original_curve.getData()[1], expected.spectrum.original
            )
            seen[name] = [
                line.value() for line in renderer.cutoff_lines if line.isVisible()
            ]
    assert seen["PyQtGraph"] == seen["Matplotlib"] == [30.0]


def test_window_goes_stale_when_selection_or_section_changes(opened, wait_for_signal):
    viewer, _ = opened
    window = _select_and_compare(viewer, (1.0, 2.0))
    assert window.aggregate is not None

    ctrl_click(viewer, 3.0)  # selection changed after the aggregate was computed
    assert window.aggregate is None
    assert "Compare Spectra" in window._status_label.text()
    assert window.renderer.last_spectrum is None
    viewer._compare_button.click()
    assert window.aggregate is not None and window.aggregate.n_traces == 3

    viewer._next_button.click()  # section replaced
    wait_for_signal(viewer._thread.finished)
    assert window.aggregate is None
    assert viewer._compare_window is window  # still open, just empty


def test_closing_the_viewer_closes_the_comparison_window(qapp, wait_for_signal):
    viewer, _ = _open(qapp, wait_for_signal)
    window = _select_and_compare(viewer, (1.0, 2.0))
    window.close()
    assert viewer._compare_window is None  # reference dropped on close
    assert viewer.compare_indices == (0, 1)  # closing the window keeps the set

    window = _select_and_compare(viewer, (3.0, 4.0))  # now 4 traces
    assert window.aggregate is not None and window.aggregate.n_traces == 4
    viewer.close()
    assert not window.isVisible()
    assert viewer._compare_window is None


def test_comparison_window_knows_no_infrastructure_or_science():
    lines = [
        l
        for l in inspect.getsource(window_module).splitlines()
        if l.startswith(("import ", "from "))
    ]
    assert not any(
        "infrastructure" in l
        or "segyio" in l
        or "h5py" in l
        or "sqlalchemy" in l
        or "numpy" in l
        or "scipy" in l
        or "viewer_workers" in l
        or "spectrum_window import SpectrumWindow" in l
        for l in lines
    )
    assert "rfft" not in inspect.getsource(window_module)
