import numpy as np
import pytest

from giecar_seismic.application.seismic_viewer import ViewerContext
from giecar_seismic.domain.job import FilterType, Job
from giecar_seismic.ui.main_window import JOBS_TABLE_HEADERS, MainWindow
from giecar_seismic.ui.seismic_viewer import SeismicViewer
from tests.unit.test_main_window import (
    FakeTraceReader,
    FakeTraceWriter,
    _build_service,
    _dataset,
    _wait_settled,
)
from tests.unit.test_seismic_renderers import FakeViewerService


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


def test_region_qc_shows_the_jobs_filter_type_and_cutoffs_and_toggles_use_cache(
    qapp, wait_for_signal, monkeypatch
):
    # The persisted job returned with the viewer context is authoritative
    # for the cutoff markers of the regional spectrum.
    band = Job(1, 10, 4, filter_type=FilterType.BAND_PASS, upper_cutoff_hz=40)

    class BandService(FakeViewerService):
        def context(self, target):
            return ViewerContext(dataset=_dataset(), job=band)

    service = BandService()
    viewer = SeismicViewer(service, lambda d: True, target=7)
    for _ in range(2):
        wait_for_signal(viewer._thread.finished)
    try:
        assert "Band-pass" in viewer.windowTitle()
        for coordinate in (1.0, 4.0):
            viewer.renderer.click_at_coordinate(coordinate)
            while viewer._thread is not None:
                wait_for_signal(viewer._thread.finished)
        raw = viewer.regional_spectrum
        assert raw is not None and raw.job is band
        markers = viewer.spectrum_renderer.last_spectrum.cutoff_markers
        assert [(m.label, m.frequency_hz) for m in markers] == [
            ("Low cutoff", 10.0),
            ("High cutoff", 40.0),
        ]
        viewer._open_spectrum_button.click()
        assert "Band-pass" in viewer._region_window._metadata_label.text()

        def forbidden(*args, **kwargs):
            pytest.fail("presentation change performed scientific computation or I/O")

        for name in ("context", "load_section", "regional_spectrum"):
            monkeypatch.setattr(service, name, forbidden)
        monkeypatch.setattr(np.fft, "rfft", forbidden)
        monkeypatch.setattr(viewer, "_start_worker", forbidden)
        for scale in ("dB", "Linear", "dB"):
            viewer._spectrum_scale_combo.setCurrentText(scale)
            viewer._response_checkbox.setChecked(True)
            displayed = viewer.spectrum_renderer.last_spectrum
            assert displayed.scale.value == scale
            assert displayed.show_filter_response
            viewer._renderer_combo.setCurrentText("Matplotlib")
            assert viewer.spectrum_renderer.last_spectrum is displayed
            assert viewer._region_window.regional is raw
            viewer._renderer_combo.setCurrentText("PyQtGraph")
        assert viewer.regional_spectrum is raw
    finally:
        viewer.close()
