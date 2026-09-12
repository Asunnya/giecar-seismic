"""SeismicViewer dialog and its workers, headless, against a fake service.
Loads run on real QThreads; tests synchronize on thread.finished.
"""

import inspect
import threading

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
from giecar_seismic.ui import seismic_viewer as viewer_module
from giecar_seismic.ui import viewer_workers as workers_module
from giecar_seismic.ui.seismic_renderer import amplitude_limit, wiggle_stride
from giecar_seismic.ui.seismic_viewer import SeismicViewer
from giecar_seismic.ui.viewer_workers import GeometryIndexWorker, SectionLoadWorker

N_SAMPLES = 32
INLINES = [10, 11, 12]
CROSSLINES = [1, 2, 3]


def _dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="s",
        source_path="/s.segy",
        n_inlines=3,
        n_crosslines=3,
        n_traces=8,
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
    physical = (
        np.asarray([3, 4, -1]) if line == 11 else np.arange(len(axis))
    )  # (11, 3) missing
    rng = np.random.default_rng(line)
    original = rng.standard_normal((len(axis), N_SAMPLES)).astype(np.float32)
    filtered = (original * 0.5).astype(np.float32)
    original[physical < 0] = np.nan
    filtered[physical < 0] = np.nan
    return SeismicSection(orientation, line, axis, physical, original, filtered, 4.0)


class FakeViewerService:
    """Records every load so tests can prove exactly which lines were
    requested -- and that nothing resembling the volume was."""

    def __init__(self):
        self.load_calls: list[tuple[LineOrientation, int]] = []
        self.load_thread_names: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def context(self, job_id):
        return ViewerContext(dataset=_dataset(), job=_job())

    def line_numbers(self, dataset_id, orientation):
        return INLINES if orientation is LineOrientation.INLINE else CROSSLINES

    def load_section(self, job_id, orientation, line_number):
        self.load_calls.append((orientation, line_number))
        self.load_thread_names.append(threading.current_thread().name)
        self.entered.set()
        assert self.release.wait(timeout=5), "test deadlocked"
        return _section(orientation, line_number)

    def select_trace(self, job_id, section, coordinate):
        position = int(np.argmin(np.abs(section.coordinates - coordinate)))
        idx = int(section.physical_trace_indices[position])
        if idx < 0:
            return None
        return TraceView(
            geometry=TraceGeometry(idx, 99, int(section.coordinates[position])),
            time_ms=section.time_ms,
            original=section.original[position],
            filtered=section.filtered[position],
            dataset=_dataset(),
            job=_job(),
        )

    def spectrum(self, view):
        f = np.fft.rfftfreq(N_SAMPLES, d=0.004)
        return TraceSpectrum(
            f,
            np.abs(np.fft.rfft(view.original)),
            np.abs(np.fft.rfft(view.filtered)),
            125.0,
            30.0,
        )


def _open(qapp, wait_for_signal, build_index=None):
    service = FakeViewerService()
    index_calls: list[SeismicDataset] = []

    def default_build(dataset):
        index_calls.append(dataset)
        return True

    viewer = SeismicViewer(service, build_index or default_build, job_id=7)
    # geometry index thread, then first section thread
    for _ in range(2):
        thread = viewer._thread
        assert thread is not None
        wait_for_signal(thread.finished)
    return viewer, service, index_calls


# --- workers ---------------------------------------------------------------


def test_viewer_workers_never_import_widgets_or_infrastructure():
    lines = [
        l
        for l in inspect.getsource(workers_module).splitlines()
        if l.startswith(("import ", "from "))
    ]
    assert not any(
        "QtWidgets" in l or "infrastructure" in l or "sqlalchemy" in l for l in lines
    )


def test_geometry_index_worker_emits_finished_or_failed():
    finished: list[bool] = []
    failed: list[str] = []
    ok = GeometryIndexWorker(lambda d: True, _dataset())
    ok.finished.connect(finished.append)
    ok.run()

    def boom(_d):
        raise OSError("bad headers")

    bad = GeometryIndexWorker(boom, _dataset())
    bad.failed.connect(failed.append)
    bad.run()

    assert finished == [True] and failed == ["bad headers"]


def test_section_load_worker_emits_the_section():
    service = FakeViewerService()
    loaded: list[SeismicSection] = []
    worker = SectionLoadWorker(service, 7, LineOrientation.CROSSLINE, 2)
    worker.loaded.connect(loaded.append)

    worker.run()

    assert service.load_calls == [(LineOrientation.CROSSLINE, 2)]
    assert loaded[0].orientation is LineOrientation.CROSSLINE


# --- dialog ------------------------------------------------------------------


def test_viewer_indexes_geometry_then_loads_the_first_inline_off_the_gui_thread(
    qapp, wait_for_signal
):
    viewer, service, index_calls = _open(qapp, wait_for_signal)

    assert [d.id for d in index_calls] == [1]
    assert service.load_calls == [(LineOrientation.INLINE, 10)]
    assert service.load_thread_names != [threading.current_thread().name]
    assert viewer._section is not None
    assert viewer._line_spinbox.value() == 10
    assert "Inline 10" in viewer._status_label.text()
    assert viewer._thread is None and viewer._worker is None


def test_navigation_is_disabled_while_a_load_is_in_flight(qapp, wait_for_signal):
    viewer, service, _ = _open(qapp, wait_for_signal)
    service.release.clear()
    service.entered.clear()

    viewer._next_button.click()  # -> inline 11, worker blocks in load_section
    assert service.entered.wait(timeout=5)
    assert viewer._line_spinbox.isEnabled() is False
    assert viewer._next_button.isEnabled() is False
    assert viewer._prev_button.isEnabled() is False
    assert "Loading inline 11" in viewer._status_label.text()

    thread = viewer._thread
    assert thread is not None
    service.release.set()
    wait_for_signal(thread.finished)

    assert viewer._line_spinbox.isEnabled() is True
    assert service.load_calls[-1] == (LineOrientation.INLINE, 11)


def test_changing_line_loads_only_that_line_never_the_volume(qapp, wait_for_signal):
    viewer, service, _ = _open(qapp, wait_for_signal)

    viewer._line_spinbox.setValue(12)
    thread = viewer._thread
    assert thread is not None
    wait_for_signal(thread.finished)

    assert service.load_calls == [
        (LineOrientation.INLINE, 10),
        (LineOrientation.INLINE, 12),
    ]
    section = viewer._section
    assert section is not None
    assert section.original.shape == (len(CROSSLINES), N_SAMPLES)  # one line
    assert section.original.shape[0] < _dataset().n_traces


def test_spinbox_snaps_to_the_nearest_existing_line(qapp, wait_for_signal):
    viewer, service, _ = _open(qapp, wait_for_signal)
    service.line_numbers = lambda d, o: [10, 20, 30]  # type: ignore[method-assign]
    viewer._line_numbers = [10, 20, 30]
    viewer._line_spinbox.setRange(10, 30)

    viewer._line_spinbox.setValue(24)  # no line 24 -> 20
    thread = viewer._thread
    assert thread is not None
    wait_for_signal(thread.finished)

    assert viewer._line_spinbox.value() == 20
    assert service.load_calls[-1] == (LineOrientation.INLINE, 20)


def test_switching_orientation_loads_the_first_crossline(qapp, wait_for_signal):
    viewer, service, _ = _open(qapp, wait_for_signal)

    viewer._orientation_combo.setCurrentIndex(1)
    thread = viewer._thread
    assert thread is not None
    wait_for_signal(thread.finished)

    assert service.load_calls[-1] == (LineOrientation.CROSSLINE, 1)
    assert viewer._section is not None
    assert viewer._section.orientation is LineOrientation.CROSSLINE


def test_selecting_a_coordinate_shows_trace_metadata_and_spectrum(
    qapp, wait_for_signal
):
    viewer, _, _ = _open(qapp, wait_for_signal)

    viewer.select_coordinate(2.2)  # nearest crossline 2

    text = viewer._trace_info_label.text()
    assert "Trace 1" in text and "crossline 2" in text
    assert "Nyquist 125.0 Hz" in text
    assert "cutoff 30.0 Hz" in text and "order 4" in text
    assert len(viewer.renderer.trace_figure.axes) == 1
    spectrum_ax = viewer.renderer.spectrum_figure.axes[0]
    assert spectrum_ax.get_xlim()[1] == pytest.approx(125.0)
    assert any("cutoff 30.0" in line.get_label() for line in spectrum_ax.get_lines())


def test_selecting_a_missing_position_reports_it_instead_of_inventing_a_trace(
    qapp, wait_for_signal
):
    viewer, _, _ = _open(qapp, wait_for_signal)
    viewer._line_spinbox.setValue(11)
    thread = viewer._thread
    assert thread is not None
    wait_for_signal(thread.finished)

    viewer.select_coordinate(3.0)  # (11, 3) missing

    assert viewer._selected is None
    assert "No trace at this position" in viewer._trace_info_label.text()
    assert len(viewer.renderer.trace_figure.axes) == 0


@pytest.mark.parametrize(
    "mode,expected_panels",
    [("Original", 1), ("Filtered", 1), ("Difference", 1), ("Side-by-side", 2)],
)
def test_display_modes_draw_the_expected_number_of_panels(
    qapp, wait_for_signal, mode, expected_panels
):
    viewer, _, _ = _open(qapp, wait_for_signal)

    viewer._mode_combo.setCurrentText(mode)

    axes = viewer.renderer.section_figure.axes
    assert len(axes) == expected_panels
    assert all(ax.get_ylabel() == "Time (ms)" for ax in axes)
    assert all(ax.get_xlabel() == "Crossline" for ax in axes)
    assert all(
        ax.get_ylim()[0] > ax.get_ylim()[1] for ax in axes
    )  # time increases downwards
    if mode == "Side-by-side":
        assert (
            axes[0].images[0].get_clim() == axes[1].images[0].get_clim()
        )  # shared scale


def test_wiggle_mode_draws_lines_instead_of_an_image(qapp, wait_for_signal):
    viewer, _, _ = _open(qapp, wait_for_signal)

    viewer._wiggle_checkbox.setChecked(True)

    ax = viewer.renderer.section_figure.axes[0]
    assert len(ax.images) == 0
    assert len(ax.get_lines()) == len(CROSSLINES)
    assert ax.get_ylim()[0] > ax.get_ylim()[1]


def test_close_is_rejected_while_a_load_is_running(qapp, wait_for_signal):
    from PyQt5.QtGui import QCloseEvent

    viewer, service, _ = _open(qapp, wait_for_signal)
    service.release.clear()
    service.entered.clear()
    viewer._next_button.click()
    assert service.entered.wait(timeout=5)

    event = QCloseEvent()
    viewer.closeEvent(event)
    assert event.isAccepted() is False

    thread = viewer._thread
    assert thread is not None
    service.release.set()
    wait_for_signal(thread.finished)
    idle = QCloseEvent()
    viewer.closeEvent(idle)
    assert idle.isAccepted() is True


def test_viewer_module_imports_no_infrastructure():
    lines = [
        l
        for l in inspect.getsource(viewer_module).splitlines()
        if l.startswith(("import ", "from "))
    ]
    assert not any(
        "infrastructure" in l or "sqlalchemy" in l or "h5py" in l or "segyio" in l
        for l in lines
    )


# --- pure display helpers ----------------------------------------------------


def test_amplitude_limit_is_shared_percentile_over_present_samples_divided_by_gain():
    section = _section(line=11)  # has a NaN gap
    limit_gain1 = amplitude_limit(section, 100.0, 1.0)
    present = np.concatenate(
        [
            section.original[section.present_mask].ravel(),
            section.filtered[section.present_mask].ravel(),
        ]
    )
    assert limit_gain1 == pytest.approx(np.abs(present).max())
    assert amplitude_limit(section, 100.0, 2.0) == pytest.approx(limit_gain1 / 2)
    assert np.isfinite(limit_gain1)


def test_wiggle_stride_decimates_only_beyond_the_display_budget():
    assert wiggle_stride(50) == 1
    assert wiggle_stride(200) == 1
    assert wiggle_stride(201) == 2
    assert wiggle_stride(1000) == 5
