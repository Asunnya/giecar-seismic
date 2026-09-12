from dataclasses import replace

import numpy as np
import pytest

from giecar_seismic.application.seismic_viewer import (
    SeismicViewerService,
    SpectrumScale,
    spectrum_for_display,
)
from giecar_seismic.domain.job import FilterType, Job
from giecar_seismic.ui.main_window import JOBS_TABLE_HEADERS, MainWindow
from giecar_seismic.ui.matplotlib_renderer import MatplotlibSeismicRenderer
from giecar_seismic.ui.pyqtgraph_renderer import PyQtGraphSeismicRenderer
from tests.unit.test_main_window import (
    FakeTraceReader,
    FakeTraceWriter,
    _build_service,
    _dataset,
    _wait_settled,
)
from tests.unit.test_seismic_renderers import _open
from tests.unit.test_spectrum_science import make_view


@pytest.mark.parametrize(
    "kind,label,upper_visible",
    [
        (FilterType.LOW_PASS, "High cutoff (Hz):", False),
        (FilterType.HIGH_PASS, "Low cutoff (Hz):", False),
        (FilterType.BAND_PASS, "Low cutoff (Hz):", True),
    ],
)
def test_configuration_controls_and_nyquist(qapp, kind, label, upper_visible):
    window = MainWindow()
    try:
        window.set_dataset(_dataset())
        window._filter_type_combo.setCurrentIndex(list(FilterType).index(kind))
        assert window._cutoff_label.text() == label
        assert window._upper_cutoff_spinbox.isHidden() is not upper_visible
        assert window._upper_cutoff_label.isHidden() is not upper_visible
        assert window._cutoff_spinbox.maximum() < 125
        assert window._upper_cutoff_spinbox.maximum() < 125
        if upper_visible:
            # low < high is not a widget bound (that would block typing a
            # 3-digit low): values stay as set, Run/Preview are gated and
            # create_filter_job() remains the authority.
            window._cutoff_spinbox.setValue(124.99)
            assert window._cutoff_spinbox.value() == pytest.approx(124.99)
            assert window._upper_cutoff_spinbox.value() == pytest.approx(40.0)
            assert not window._run_button.isEnabled()
            window._upper_cutoff_spinbox.setValue(0.01)
            assert window._upper_cutoff_spinbox.value() == pytest.approx(0.01)
            assert not window._run_button.isEnabled()
            window._cutoff_spinbox.setValue(0.005)  # rounds to the 0.01 floor
            window._upper_cutoff_spinbox.setValue(0.02)
            assert window._cutoff_spinbox.value() == pytest.approx(0.01)
    finally:
        window.close()


@pytest.mark.parametrize(
    "kind,upper,expected",
    [
        (FilterType.LOW_PASS, None, "10 Hz"),
        (FilterType.HIGH_PASS, None, "10 Hz"),
        (FilterType.BAND_PASS, 40, "10–40 Hz"),
    ],
)
def test_main_window_creates_configured_job_and_history(
    qapp, wait_for_signal, kind, upper, expected
):
    dataset = _dataset()
    service = _build_service(
        dataset, FakeTraceReader(np.ones((4, 64))), FakeTraceWriter()
    )
    window = MainWindow(service=service)
    try:
        while window._history_thread is not None:
            wait_for_signal(window._history_thread.finished)
        window.set_dataset(dataset)
        window._filter_type_combo.setCurrentIndex(list(FilterType).index(kind))
        window._cutoff_spinbox.setValue(10)
        if upper:
            window._upper_cutoff_spinbox.setValue(upper)
        window._run_button.click()
        _wait_settled(window, wait_for_signal, window._thread.finished)
        job = service.list_jobs()[0]
        assert job.filter_type is kind
        assert job.upper_cutoff_hz == upper
        table = window._jobs_table
        assert (
            table.item(0, JOBS_TABLE_HEADERS.index("Filter")).text()
            == {
                FilterType.LOW_PASS: "Low-pass",
                FilterType.HIGH_PASS: "High-pass",
                FilterType.BAND_PASS: "Band-pass",
            }[kind]
        )
        assert table.item(0, JOBS_TABLE_HEADERS.index("Cutoff(s)")).text() == expected
        assert table.item(0, JOBS_TABLE_HEADERS.index("Created at")).text()
    finally:
        window.close()


@pytest.mark.parametrize(
    "kind,upper",
    [
        (FilterType.LOW_PASS, None),
        (FilterType.HIGH_PASS, None),
        (FilterType.BAND_PASS, 50),
    ],
)
@pytest.mark.parametrize("scale", list(SpectrumScale))
def test_renderers_draw_identical_scientific_data(qapp, kind, upper, scale):
    view = make_view(kind, 15, upper)
    data = spectrum_for_display(
        SeismicViewerService.spectrum(view), scale, show_filter_response=True
    )
    mpl, pg = MatplotlibSeismicRenderer(), PyQtGraphSeismicRenderer()
    try:
        for renderer in (mpl, pg):
            renderer.show_trace(view, data)
            assert renderer.last_spectrum is data
        ax = mpl.spectrum_figure.axes[0]
        assert ax.get_xscale() == "linear"
        assert ax.get_xlim() == (0, 125)
        assert ax.get_ylabel() == data.magnitude_label
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), data.original)
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), data.filtered)
        markers = ax.lines[2:]
        assert len(markers) == len(data.cutoff_markers)
        for line, marker in zip(markers, data.cutoff_markers, strict=True):
            assert line.get_xdata()[0] == marker.frequency_hz
            assert marker.label in line.get_label()
        response_ax = mpl.spectrum_figure.axes[1]
        np.testing.assert_array_equal(
            response_ax.lines[0].get_ydata(), data.filter_response
        )
        assert response_ax.get_ylabel() == data.response_label
        np.testing.assert_array_equal(pg._spectrum_original.getData()[1], data.original)
        np.testing.assert_array_equal(pg._spectrum_filtered.getData()[1], data.filtered)
        np.testing.assert_array_equal(
            pg._response_curve.getData()[1], data.filter_response
        )
        visible = [line for line in pg._cutoff_lines if line.isVisible()]
        assert [line.value() for line in visible] == [
            m.frequency_hz for m in data.cutoff_markers
        ]
        for line, marker in zip(visible, data.cutoff_markers, strict=True):
            assert (
                line.label.toPlainText() == f"{marker.label} {marker.frequency_hz} Hz"
            )
        assert pg._spectrum_plot.getAxis("left").labelText == data.magnitude_label
        assert pg._spectrum_plot.getViewBox().viewRange()[0] == [0, 125]
        assert not pg._spectrum_plot.getPlotItem().ctrl.logXCheck.isChecked()
        for renderer in (mpl, pg):
            renderer.show_trace(
                view, spectrum_for_display(SeismicViewerService.spectrum(view), scale)
            )
        assert len(mpl.spectrum_figure.axes) == 1
        assert not pg._response_curve.isVisible()
        mpl.show_trace(None, None)
        pg.show_trace(None, None)
        assert not any(line.isVisible() for line in pg._cutoff_lines)
    finally:
        mpl.dispose()
        pg.dispose()


def test_scale_response_and_renderer_toggles_use_cached_spectrum(
    qapp, wait_for_signal, monkeypatch
):
    viewer, service = _open(qapp, wait_for_signal)
    try:
        # The persisted job returned with the trace is authoritative.
        select = service.select_trace

        def select_band(*args):
            view = select(*args)
            return replace(
                view,
                job=Job(1, 10, 4, filter_type=FilterType.BAND_PASS, upper_cutoff_hz=40),
            )

        monkeypatch.setattr(service, "select_trace", select_band)
        monkeypatch.setattr(service, "spectrum", SeismicViewerService.spectrum)
        viewer.select_coordinate(2)
        text = viewer._trace_info_label.text()
        assert (
            "Band-pass" in text and "Low cutoff 10" in text and "High cutoff 40" in text
        )
        raw = viewer._selected_spectrum

        def forbidden(*args, **kwargs):
            pytest.fail("presentation change performed scientific computation or I/O")

        for name in ("context", "load_section", "select_trace", "spectrum"):
            monkeypatch.setattr(service, name, forbidden)
        monkeypatch.setattr(np.fft, "rfft", forbidden)
        monkeypatch.setattr(viewer, "_start_worker", forbidden)
        for scale in ("dB", "Linear", "dB"):
            viewer._spectrum_scale_combo.setCurrentText(scale)
            viewer._response_checkbox.setChecked(True)
            displayed = viewer.renderer.last_spectrum
            assert displayed.scale.value == scale
            assert displayed.show_filter_response
            viewer._renderer_combo.setCurrentText("Matplotlib")
            assert viewer.renderer.last_spectrum is displayed
            viewer._renderer_combo.setCurrentText("PyQtGraph")
            assert viewer.renderer.last_spectrum is displayed
            assert viewer._selected_spectrum is raw
            assert viewer._thread is None
    finally:
        viewer.close()
