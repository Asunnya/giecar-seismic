"""Modeless spectrum: data identity, no acquisition, synchronized state and lifetime."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, Qt

from giecar_seismic.application.seismic_viewer import (
    SeismicViewerService,
    SpectrumScale,
)
from giecar_seismic.domain.job import FilterType
from tests.unit.test_seismic_renderers import _open, _section


@pytest.fixture
def opened(qapp, wait_for_signal, monkeypatch):
    viewer, service = _open(qapp, wait_for_signal)
    monkeypatch.setattr(service, "spectrum", SeismicViewerService.spectrum)
    yield viewer, service
    if viewer._thread is not None:
        service.release.set()
        wait_for_signal(viewer._thread.finished)
    viewer.close()


def select_and_open(viewer):
    viewer.select_coordinate(2)
    viewer._open_spectrum_button.click()
    return viewer._spectrum_window


def test_open_is_disabled_until_data_exists_then_reuses_modeless_window(opened):
    viewer, _ = opened
    assert not viewer._open_spectrum_button.isEnabled()
    viewer._open_spectrum_button.click()
    assert viewer._spectrum_window is None
    window = select_and_open(viewer)
    assert viewer._open_spectrum_button.isEnabled()
    assert window.isVisible()
    assert not window.isModal()
    assert window.windowModality() == Qt.NonModal
    assert window.width() >= 900 and window.height() >= 600
    viewer._open_spectrum_button.click()
    assert viewer._spectrum_window is window
    assert window.spectrum is viewer._display_spectrum
    assert window.renderer.last_spectrum is viewer.renderer.last_spectrum


def test_open_and_inspection_never_request_data_or_repeat_fft(opened, monkeypatch):
    viewer, service = opened
    viewer.select_coordinate(2)
    data = viewer._display_spectrum

    def forbidden(*args, **kwargs):
        pytest.fail("spectrum presentation called acquisition/science")

    for name in ("context", "line_numbers", "load_section", "select_trace", "spectrum"):
        monkeypatch.setattr(service, name, forbidden)
    monkeypatch.setattr(viewer, "_start_worker", forbidden)
    monkeypatch.setattr(np.fft, "rfft", forbidden)
    import sqlite3

    import h5py
    import segyio

    monkeypatch.setattr(h5py, "File", forbidden)
    monkeypatch.setattr(segyio, "open", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    import giecar_seismic.application.seismic_viewer as science

    monkeypatch.setattr(science, "butterworth_sos", forbidden)
    monkeypatch.setattr(science, "sosfreqz", forbidden)
    viewer._open_spectrum_button.click()
    window = viewer._spectrum_window
    assert window.spectrum is data
    for name in ("Matplotlib", "PyQtGraph"):
        window._renderer_combo.setCurrentText(name)
        assert window.renderer.last_spectrum is data
    window._original_checkbox.setChecked(False)
    window._filtered_checkbox.setChecked(False)
    assert "No curves selected" in window._status_label.text()
    window._response_checkbox.setChecked(True)
    assert window.renderer.visibility.response
    window._scale_combo.setCurrentText("dB")
    assert viewer._display_spectrum is window.spectrum
    assert window.spectrum.scale is SpectrumScale.DB
    assert viewer._thread is None


def test_scale_is_owned_by_viewer_and_synchronized_both_ways(opened):
    viewer, _ = opened
    window = select_and_open(viewer)
    window._original_checkbox.setChecked(False)
    window._response_checkbox.setChecked(True)
    window._scale_combo.setCurrentText("dB")
    assert viewer._spectrum_scale_combo.currentText() == "dB"
    assert viewer.renderer.last_spectrum is window.renderer.last_spectrum
    assert window.spectrum.scale is SpectrumScale.DB
    assert not viewer._response_checkbox.isChecked()  # visibility stays local
    viewer._spectrum_scale_combo.setCurrentText("Linear")
    assert window._scale_combo.currentText() == "Linear"
    assert window.spectrum.scale is SpectrumScale.LINEAR
    window._renderer_combo.setCurrentText("Matplotlib")
    assert not window._original_checkbox.isChecked()
    assert window._response_checkbox.isChecked()
    assert not window.renderer.visibility.original
    assert window.renderer.visibility.response


@pytest.mark.parametrize(
    "kind,upper,expected",
    [
        (FilterType.LOW_PASS, None, ["High cutoff"]),
        (FilterType.HIGH_PASS, None, ["Low cutoff"]),
        (FilterType.BAND_PASS, 50, ["Low cutoff", "High cutoff"]),
    ],
)
def test_live_trace_metadata_and_cutoffs_come_from_viewer(
    opened, monkeypatch, kind, upper, expected
):
    viewer, service = opened
    select = service.select_trace

    def selected(*args):
        view = select(*args)
        return replace(
            view, job=replace(view.job, filter_type=kind, upper_cutoff_hz=upper)
        )

    monkeypatch.setattr(service, "select_trace", selected)
    window = select_and_open(viewer)
    first = window.spectrum
    assert "Job 7" in window._identity_label.text()
    assert "Dataset: s" in window._identity_label.text()
    assert "Inline 10" in window._identity_label.text()
    assert "Crossline 2" in window._identity_label.text()
    assert "Physical trace index 1" in window._identity_label.text()
    assert "Order: 4" in window._metadata_label.text()
    assert "Sample rate: 4.0 ms" in window._metadata_label.text()
    assert "Nyquist: 125" in window._metadata_label.text()
    assert [m.label for m in window.spectrum.cutoff_markers] == expected
    viewer.select_coordinate(3)
    assert window.spectrum is viewer._display_spectrum
    assert window.spectrum is not first
    assert "Crossline 3" in window._identity_label.text()
    assert "Physical trace index 2" in window._identity_label.text()


def test_navigation_blocks_stale_selection_and_late_result(opened, wait_for_signal):
    viewer, service = opened
    window = select_and_open(viewer)
    previous_request = viewer._section_request_id
    service.release.clear()
    service.entered.clear()
    viewer._next_button.click()
    assert service.entered.wait(5)
    viewer.select_coordinate(1)  # old section remains drawn, but is not selectable
    assert window.spectrum is None
    assert not viewer._open_spectrum_button.isEnabled()
    service.release.set()
    wait_for_signal(viewer._thread.finished)
    viewer.select_coordinate(2)
    current = window.spectrum
    assert "Inline 11" in window._identity_label.text()
    viewer._accept_section_result(previous_request, _section(line=10))
    assert window.spectrum is current
    # A duplicate result from the already consumed request is stale too.
    viewer._accept_section_result(viewer._section_request_id, _section(line=11))
    assert window.spectrum is current
    viewer.select_coordinate(3)  # missing geometry in inline 11
    assert window.spectrum is None
    assert "Select a seismic trace" in window._status_label.text()


def test_close_reopen_disconnects_signals_and_releases_renderers(opened, monkeypatch):
    viewer, _ = opened
    render = Mock(wraps=viewer.renderer.show_trace)
    monkeypatch.setattr(viewer.renderer, "show_trace", render)
    viewer.select_coordinate(2)
    for _ in range(3):
        viewer._open_spectrum_button.click()
        window = viewer._spectrum_window
        renderer = window.renderer
        widget = renderer.widget()
        before = render.call_count
        window._scale_combo.setCurrentText("dB")
        assert render.call_count == before + 1
        window.close()
        assert viewer._spectrum_window is None
        assert renderer.disposed
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(widget)
        assert sip.isdeleted(window)
        viewer._spectrum_scale_combo.setCurrentText("Linear")
        render.reset_mock()
    viewer._open_spectrum_button.click()
    window = viewer._spectrum_window
    viewer.close()
    assert viewer._spectrum_window is None
    assert window.renderer.disposed


def test_escape_closes_only_spectrum_window(opened):
    viewer, _ = opened
    window = select_and_open(viewer)
    window.reject()
    assert viewer._spectrum_window is None
    assert window.renderer.disposed
    assert viewer._renderer is not None


def test_switching_renderer_disposes_old_widget_and_preserves_controls(opened):
    viewer, _ = opened
    window = select_and_open(viewer)
    window._scale_combo.setCurrentText("dB")
    window._filtered_checkbox.setChecked(False)
    window._response_checkbox.setChecked(True)
    metadata = window._metadata_label.text()
    for name in ("Matplotlib", "PyQtGraph"):
        old = window.renderer
        widget = old.widget()
        window._renderer_combo.setCurrentText(name)
        assert old.disposed
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(widget)
        assert window.spectrum is viewer._display_spectrum
        assert window._scale_combo.currentText() == "dB"
        assert window._original_checkbox.isChecked()
        assert not window._filtered_checkbox.isChecked()
        assert window._response_checkbox.isChecked()
        assert window._metadata_label.text() == metadata


def test_viewer_close_during_load_keeps_child_until_worker_finishes(
    opened, wait_for_signal
):
    viewer, service = opened
    window = select_and_open(viewer)
    service.release.clear()
    service.entered.clear()
    viewer._next_button.click()
    assert service.entered.wait(5)
    assert not viewer.close()
    assert viewer._spectrum_window is window
    assert not window.renderer.disposed
    service.release.set()
    wait_for_signal(viewer._thread.finished)
    viewer.reject()  # Escape must use the same cleanup as closing the viewer
    assert viewer._spectrum_window is None
    assert window.renderer.disposed


def test_scale_stays_synchronized_while_navigation_has_no_selected_trace(
    opened, wait_for_signal
):
    viewer, _ = opened
    window = select_and_open(viewer)
    viewer._next_button.click()
    wait_for_signal(viewer._thread.finished)
    assert window.spectrum is None
    viewer._spectrum_scale_combo.setCurrentText("dB")
    assert window._scale_combo.currentText() == "dB"
    window._scale_combo.setCurrentText("Linear")
    assert viewer._spectrum_scale_combo.currentText() == "Linear"
    assert window.spectrum is None
