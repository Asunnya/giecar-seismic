from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import butter, sosfreqz

from giecar_seismic.application.seismic_viewer import (
    AggregateSpectrum,
    SeismicSection,
    SeismicViewerService,
    SpectrumScale,
    TraceView,
    aggregate_for_display,
    spectrum_for_display,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation, TraceGeometry
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


# --- multi-trace aggregate spectrum ------------------------------------------
#
# Ensemble QC: mean of per-trace amplitude spectra, TRACE -> FFT -> |.| ->
# mean over traces. Never |FFT(mean of traces)|: time-domain averaging lets
# neighbouring traces cancel by phase and misrepresents the frequency content.

N_AGG = 1000
FS_HZ = 250.0
DATASET = SeismicDataset("s", "/s.segy", 1, 5, 5, N_AGG, 4.0)
JOB = Job(1, 30.0, 4)


def make_section(traces: np.ndarray, *, physical=None, line=10) -> SeismicSection:
    """Inline `line` with axis crosslines 1..n; `physical[i] = -1` is a gap."""
    n = traces.shape[0]
    physical = np.arange(n) if physical is None else np.asarray(physical)
    original = traces.astype(np.float32)
    filtered = (original * 0.1).astype(np.float32)
    original[physical < 0] = np.nan
    filtered[physical < 0] = np.nan
    return SeismicSection(
        LineOrientation.INLINE,
        line,
        np.arange(1, n + 1),
        physical,
        original,
        filtered,
        4.0,
    )


def _sines(frequencies_hz, phases):
    t = np.arange(N_AGG) / FS_HZ
    return np.array(
        [np.sin(2 * np.pi * f * t + p) for f, p in zip(frequencies_hz, phases)]
    )


def test_position_for_coordinate_snaps_to_nearest_axis_position_or_none():
    section = make_section(_sines([10, 20, 30], [0, 0, 0]))  # axis 1, 2, 3
    assert section.position_for_coordinate(2.3) == 1
    assert section.position_for_coordinate(1.0) == 0
    assert section.position_for_coordinate(3.49) == 2
    assert section.position_for_coordinate(4.2) is None  # outside the axis
    empty = replace(section, coordinates=np.array([], dtype=np.int64))
    assert empty.position_for_coordinate(1.0) is None


def test_aggregate_is_the_mean_of_per_trace_amplitude_spectra():
    section = make_section(_sines([10, 20, 40], [0, 0.3, 1.1]))

    aggregate = SeismicViewerService.aggregate_spectrum(section, [2, 0], DATASET, JOB)

    per_trace_original = np.abs(np.fft.rfft(section.original[[2, 0]], axis=-1))
    per_trace_filtered = np.abs(np.fft.rfft(section.filtered[[2, 0]], axis=-1))
    np.testing.assert_allclose(
        aggregate.spectrum.original, per_trace_original.mean(axis=0), rtol=1e-6
    )
    np.testing.assert_allclose(
        aggregate.spectrum.filtered, per_trace_filtered.mean(axis=0), rtol=1e-6
    )
    assert aggregate.trace_indices == (2, 0)  # selection order, physical identity
    assert aggregate.coordinates == (3, 1)
    assert aggregate.n_traces == 2
    assert aggregate.orientation is LineOrientation.INLINE
    assert aggregate.line_number == 10
    assert aggregate.dataset is DATASET and aggregate.job is JOB
    assert "Mean of per-trace amplitude spectra" in AggregateSpectrum.METHOD


def test_aggregate_is_not_the_spectrum_of_the_averaged_traces():
    # Two traces of the same 20 Hz sine in anti-phase: their time-domain
    # mean is ~0 everywhere, but each has a strong 20 Hz line.
    section = make_section(_sines([20, 20], [0, np.pi]))

    aggregate = SeismicViewerService.aggregate_spectrum(section, [0, 1], DATASET, JOB)

    wrong = np.abs(np.fft.rfft(section.original[[0, 1]].mean(axis=0)))
    peak = np.argmin(np.abs(aggregate.spectrum.frequencies_hz - 20))
    assert wrong[peak] < 1e-3 * aggregate.spectrum.original[peak]  # cancelled
    assert aggregate.spectrum.original[peak] > 100  # preserved per-trace energy
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(aggregate.spectrum.original, wrong, rtol=1e-3)


def test_aggregate_axis_markers_and_single_response_match_the_single_trace_path():
    band = Job(1, 15.0, 4, filter_type=FilterType.BAND_PASS, upper_cutoff_hz=50.0)
    section = make_section(_sines([10, 20, 40, 60], [0, 1, 2, 3]))

    aggregate = SeismicViewerService.aggregate_spectrum(
        section, [0, 1, 2, 3], DATASET, band
    )
    single = SeismicViewerService.spectrum(
        TraceView(
            TraceGeometry(0, 10, 1),
            section.time_ms,
            section.original[0],
            section.filtered[0],
            DATASET,
            band,
        )
    )

    spectrum = aggregate.spectrum
    np.testing.assert_array_equal(
        spectrum.frequencies_hz, np.fft.rfftfreq(N_AGG, 0.004)
    )
    np.testing.assert_array_equal(spectrum.frequencies_hz, single.frequencies_hz)
    assert spectrum.nyquist_hz == 125.0
    assert spectrum.filter_type is FilterType.BAND_PASS
    assert (spectrum.cutoff_hz, spectrum.upper_cutoff_hz) == (15.0, 50.0)
    assert [m.label for m in spectrum.cutoff_markers] == ["Low cutoff", "High cutoff"]
    # one theoretical zero-phase response for the aggregate axis -- the same
    # curve as the single-trace path, not N responses averaged
    np.testing.assert_array_equal(spectrum.filter_response, single.filter_response)
    assert spectrum.filter_response.shape == spectrum.frequencies_hz.shape
    assert spectrum.scale is SpectrumScale.LINEAR


def test_db_is_applied_after_linear_aggregation_with_the_shared_rules():
    section = make_section(_sines([20, 20, 40], [0, 0.5, 1.0]))
    aggregate = SeismicViewerService.aggregate_spectrum(
        section, [0, 1, 2], DATASET, JOB
    )

    db = aggregate_for_display(aggregate, SpectrumScale.DB, show_filter_response=True)

    linear = aggregate.spectrum
    peak = np.argmax(linear.original)
    # the reference is the *aggregate* original peak (0 dB), floor -120 dB,
    # exactly what spectrum_for_display does for a single trace
    expected = spectrum_for_display(linear, SpectrumScale.DB, show_filter_response=True)
    np.testing.assert_array_equal(db.spectrum.original, expected.original)
    np.testing.assert_array_equal(db.spectrum.filtered, expected.filtered)
    np.testing.assert_array_equal(db.spectrum.filter_response, expected.filter_response)
    assert db.spectrum.original[peak] == pytest.approx(0)
    assert db.spectrum.filtered[peak] == pytest.approx(-20)
    assert db.spectrum.reference_magnitude == pytest.approx(linear.original[peak])
    assert db.spectrum.scale is SpectrumScale.DB and db.spectrum.show_filter_response
    # NOT the mean of per-trace dB spectra: dB is not linear in magnitude
    per_trace_db = [
        spectrum_for_display(
            SeismicViewerService.spectrum(
                TraceView(
                    TraceGeometry(i, 10, i + 1),
                    section.time_ms,
                    section.original[i],
                    section.filtered[i],
                    DATASET,
                    JOB,
                )
            ),
            SpectrumScale.DB,
        ).original
        for i in range(3)
    ]
    wrong = np.mean(per_trace_db, axis=0)
    bin_40 = np.argmin(np.abs(linear.frequencies_hz - 40))
    # aggregate: (0 + 0 + A) / 3 against a peak of 2A / 3 -> about -6 dB;
    # mean of per-trace dB: two floors (-120) and one 0 dB -> about -80 dB
    assert db.spectrum.original[bin_40] == pytest.approx(-6.02, abs=0.1)
    assert wrong[bin_40] < -60
    # identity and metadata survive the display transform
    assert db.trace_indices == aggregate.trace_indices
    assert db.n_traces == 3 and db.job is JOB


def test_display_toggles_never_mutate_or_recompute_the_raw_aggregate(monkeypatch):
    section = make_section(_sines([10, 20], [0, 0]))
    aggregate = SeismicViewerService.aggregate_spectrum(section, [0, 1], DATASET, JOB)
    raw_original = aggregate.spectrum.original.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("display transform recomputed an FFT")

    monkeypatch.setattr(np.fft, "rfft", forbidden)
    db = aggregate_for_display(aggregate, SpectrumScale.DB)
    back = aggregate_for_display(aggregate, SpectrumScale.LINEAR)
    again = aggregate_for_display(
        aggregate, SpectrumScale.DB, show_filter_response=True
    )

    assert back.spectrum.original is aggregate.spectrum.original
    np.testing.assert_array_equal(aggregate.spectrum.original, raw_original)
    np.testing.assert_array_equal(again.spectrum.original, db.spectrum.original)
    assert aggregate.spectrum.scale is SpectrumScale.LINEAR
    with pytest.raises(ValueError, match="linear"):
        aggregate_for_display(db, SpectrumScale.LINEAR)  # never chain dB -> dB


def test_all_zero_traces_stay_finite_at_the_db_floor():
    section = make_section(np.zeros((2, N_AGG)))
    aggregate = SeismicViewerService.aggregate_spectrum(section, [0, 1], DATASET, JOB)
    with np.errstate(divide="raise", invalid="raise"):
        db = aggregate_for_display(
            aggregate, SpectrumScale.DB, show_filter_response=True
        )
    np.testing.assert_array_equal(db.spectrum.original, -120)
    np.testing.assert_array_equal(db.spectrum.filtered, -120)
    assert np.isfinite(db.spectrum.filter_response).all()
    assert db.spectrum.reference_magnitude == 1.0


@pytest.mark.parametrize(
    "indices,match",
    [
        ([], "at least one"),
        ([0, 0], "duplicate"),
        ([7], "not in the section"),
        ([1], "not in the section"),  # a gap: physical index -1 at that position
    ],
)
def test_aggregate_rejects_empty_duplicate_missing_and_gap_indices(indices, match):
    section = make_section(_sines([10, 20, 30], [0, 0, 0]), physical=[0, -1, 2])
    with pytest.raises(ValueError, match=match):
        SeismicViewerService.aggregate_spectrum(section, indices, DATASET, JOB)


def test_aggregate_uses_only_the_selected_rows_of_the_section():
    # bounded by the selection, not the section: untouched rows may be NaN
    traces = _sines([10, 20, 30, 40], [0, 0, 0, 0])
    section = make_section(traces, physical=[0, 1, -1, 3])
    aggregate = SeismicViewerService.aggregate_spectrum(section, [3, 0], DATASET, JOB)
    assert np.isfinite(aggregate.spectrum.original).all()
    assert aggregate.coordinates == (4, 1)
