from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import butter, sosfreqz

from giecar_seismic.application.seismic_viewer import (
    SeismicViewerService,
    SpectrumScale,
    TraceView,
    spectrum_for_display,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import TraceGeometry
from giecar_seismic.domain.job import FilterType, Job


def make_view(kind=FilterType.LOW_PASS, cutoff=30.0, upper=None, n=1000):
    t = np.arange(n) / 250
    original = np.sin(2 * np.pi * 25 * t)
    return TraceView(
        TraceGeometry(0, 1, 1),
        t * 1000,
        original,
        original * 0.1,
        SeismicDataset("s", "/s.segy", 1, 1, 1, n, 4.0),
        Job(1, cutoff, 4, filter_type=kind, upper_cutoff_hz=upper),
    )


def test_linear_fft_and_shared_db_reference():
    view = make_view()
    raw = SeismicViewerService.spectrum(view)
    linear = spectrum_for_display(raw, SpectrumScale.LINEAR)
    db = spectrum_for_display(raw, SpectrumScale.DB)
    np.testing.assert_allclose(linear.original, np.abs(np.fft.rfft(view.original)))
    assert linear.original is raw.original
    peak = np.argmax(raw.original)
    assert db.original[peak] == pytest.approx(0)
    assert db.filtered[peak] == pytest.approx(-20)
    assert db.reference_magnitude == pytest.approx(raw.original.max())
    assert db.frequencies_hz is raw.frequencies_hz
    assert (db.frequencies_hz[0], db.frequencies_hz[-1]) == (0, 125)
    assert raw.original[peak] > 1  # transformation never mutates the cached FFT


@pytest.mark.parametrize("original_scale", [0.0, 1e-250, 1.0])
def test_db_floor_is_finite_even_for_zero_spectra(original_scale):
    view = make_view()
    raw = SeismicViewerService.spectrum(
        replace(view, original=view.original * original_scale, filtered=np.zeros(1000))
    )
    db = spectrum_for_display(raw, SpectrumScale.DB, show_filter_response=True)
    assert np.isfinite(db.original).all()
    assert np.isfinite(db.filtered).all()
    assert np.isfinite(db.filter_response).all()
    np.testing.assert_array_equal(db.filtered, -120)


@pytest.mark.parametrize("n", [999, 1000])
def test_frequency_bins_remain_linear_with_nyquist_display_limit(n):
    raw = SeismicViewerService.spectrum(make_view(n=n))
    for scale in SpectrumScale:
        data = spectrum_for_display(raw, scale)
        np.testing.assert_allclose(data.frequencies_hz, np.fft.rfftfreq(n, 0.004))
        assert data.frequencies_hz[0] == 0
        assert data.frequencies_hz[-1] <= data.nyquist_hz == 125


@pytest.mark.parametrize(
    "kind,upper,labels",
    [
        (FilterType.LOW_PASS, None, ["High cutoff"]),
        (FilterType.HIGH_PASS, None, ["Low cutoff"]),
        (FilterType.BAND_PASS, 50.0, ["Low cutoff", "High cutoff"]),
    ],
)
def test_markers_and_zero_phase_response_match_scipy(kind, upper, labels):
    raw = SeismicViewerService.spectrum(make_view(kind, 15.0, upper))
    assert [m.label for m in raw.cutoff_markers] == labels
    assert [m.frequency_hz for m in raw.cutoff_markers] == (
        [15.0, 50.0] if upper else [15.0]
    )
    btype = {
        FilterType.LOW_PASS: "lowpass",
        FilterType.HIGH_PASS: "highpass",
        FilterType.BAND_PASS: "bandpass",
    }[kind]
    sos = butter(4, [15, 50] if upper else 15, btype=btype, fs=250, output="sos")
    _, h = sosfreqz(sos, worN=raw.frequencies_hz, fs=250)
    np.testing.assert_allclose(raw.filter_response, np.abs(h) ** 2)
    assert not spectrum_for_display(raw, SpectrumScale.LINEAR).show_filter_response
    db = spectrum_for_display(raw, SpectrumScale.DB, show_filter_response=True)
    np.testing.assert_allclose(
        db.filter_response, 20 * np.log10(np.maximum(np.abs(h) ** 2, 1e-6))
    )
    assert db.show_filter_response
    low, middle, high = [
        raw.filter_response[np.argmin(abs(raw.frequencies_hz - f))] for f in [3, 30, 90]
    ]
    if kind is FilterType.LOW_PASS:
        assert low > 0.95 and high < 0.05
    elif kind is FilterType.HIGH_PASS:
        assert low < 0.05 and high > 0.95
    else:
        assert low < 0.05 and middle > 0.95 and high < 0.05


def test_db_floor_handles_float32_fft_zeros_without_warnings():
    view = make_view()
    zeros = np.zeros(1000, dtype=np.float32)
    raw = SeismicViewerService.spectrum(replace(view, original=zeros, filtered=zeros))
    raw = replace(
        raw,
        original=raw.original.astype(np.float32),
        filtered=raw.filtered.astype(np.float32),
    )
    with np.errstate(divide="raise", invalid="raise"):
        db = spectrum_for_display(raw, SpectrumScale.DB)
    np.testing.assert_array_equal(db.original, -120)
    np.testing.assert_array_equal(db.filtered, -120)
