"""Regional spectral QC, application layer only (no Qt, no I/O).

A region is a contiguous interval of the loaded section's axis. Every
physically present trace inside it contributes ONE per-trace amplitude
spectrum; the regional curves are the per-bin arithmetic mean of those
spectra (TRACE -> FFT -> |.| -> MEAN), never the spectrum of averaged
traces. Aggregation is linear and chunked; dB is a display transform
applied afterwards with the same reference/floor rules for every curve.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import butter, sosfreqz

from giecar_seismic.application.seismic_viewer import (
    MIN_REGION_TRACES,
    REGION_SPECTRUM_CHUNK_TRACES,
    RegionalSpectrum,
    RegionTooSmallError,
    SectionRegion,
    SeismicSection,
    SeismicViewerService,
    SpectrumScale,
    TraceWaveform,
    regional_spectrum_for_display,
    spectrum_for_display,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.domain.job import FilterType, Job

N = 1000
FS_HZ = 250.0
DATASET = SeismicDataset("s", "/s.segy", 1, 5, 5, N, 4.0)
JOB = Job(1, 30.0, 4)


def make_section(
    traces: np.ndarray,
    *,
    physical=None,
    coordinates=None,
    line=10,
    orientation=LineOrientation.INLINE,
) -> SeismicSection:
    """A section whose axis is `coordinates` (default 1..n); `physical[i]
    = -1` is a survey gap (NaN row)."""
    n = traces.shape[0]
    physical = np.arange(n) if physical is None else np.asarray(physical)
    coordinates = (
        np.arange(1, n + 1) if coordinates is None else np.asarray(list(coordinates))
    )
    original = np.array(traces, dtype=np.float32)
    filtered = (original * 0.1).astype(np.float32)
    original[physical < 0] = np.nan
    filtered[physical < 0] = np.nan
    return SeismicSection(
        orientation, line, coordinates, physical, original, filtered, 4.0
    )


def sines(frequencies_hz, phases=None, amplitude=1.0):
    t = np.arange(N) / FS_HZ
    phases = np.zeros(len(frequencies_hz)) if phases is None else phases
    return np.array(
        [
            amplitude * np.sin(2 * np.pi * f * t + p)
            for f, p in zip(frequencies_hz, phases, strict=True)
        ]
    )


def make_regional(kind=FilterType.LOW_PASS, cutoff=30.0, upper=None, n_traces=2):
    """Reference regional spectrum for presentation tests elsewhere."""
    section = make_section(sines([25] * n_traces))
    job = Job(1, cutoff, 4, filter_type=kind, upper_cutoff_hz=upper)
    return SeismicViewerService.regional_spectrum(
        section, SectionRegion.from_boundaries(1, n_traces), DATASET, job
    )


def reference_mean_spectrum(rows: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(rows, axis=-1)).mean(axis=0)


# --- region geometry --------------------------------------------------------------


def test_region_bounds_normalize_regardless_of_click_order():
    assert SectionRegion.from_boundaries(2050, 2000) == SectionRegion(2000, 2050)
    assert SectionRegion.from_boundaries(2000, 2050) == SectionRegion(2000, 2050)
    assert SectionRegion.from_boundaries(7, 7) == SectionRegion(7, 7)


def test_region_resolution_includes_both_boundaries_and_only_present_traces():
    #  coords    2000 2001 2002 2003 2004 2005 2006 2007
    #  physical   40   41    .   43   44    .   46   47
    section = make_section(
        sines([10] * 8),
        physical=[40, 41, -1, 43, 44, -1, 46, 47],
        coordinates=range(2000, 2008),
    )

    resolved = SectionRegion.from_boundaries(2006, 2000).resolve(section)

    assert resolved.region == SectionRegion(2000, 2006)
    assert resolved.orientation is LineOrientation.INLINE
    assert resolved.line_number == 10
    assert resolved.positions == (0, 1, 2, 3, 4, 5, 6)  # inclusive, 7 axis positions
    assert resolved.present_positions == (0, 1, 3, 4, 6)
    assert resolved.trace_indices == (40, 41, 43, 44, 46)  # never 47 (outside)
    assert (resolved.n_positions, resolved.n_present, resolved.n_missing) == (7, 5, 2)
    assert resolved.is_valid


def test_region_boundary_may_sit_on_a_missing_position():
    section = make_section(sines([10] * 5), physical=[0, 1, -1, 3, 4])
    resolved = SectionRegion.from_boundaries(3, 5).resolve(section)  # 3 is a gap
    assert resolved.positions == (2, 3, 4)
    assert resolved.trace_indices == (3, 4)
    assert (resolved.n_positions, resolved.n_present, resolved.n_missing) == (3, 2, 1)


def test_region_must_sit_on_the_sections_axis():
    section = make_section(sines([10] * 4))  # axis 1..4
    with pytest.raises(ValueError, match="axis"):
        SectionRegion(1, 9).resolve(section)
    with pytest.raises(ValueError, match="axis"):
        SectionRegion(0, 2).resolve(section)


def test_single_trace_region_is_valid_and_equals_that_traces_spectrum():
    # Two clicks on the same position (or an interval with one present
    # trace) is the single-trace analysis: a mean over one spectrum.
    section = make_section(sines([10, 20, 30, 40, 50], [0, 1, 2, 3, 4]))

    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion.from_boundaries(3, 3), DATASET, JOB
    )

    assert regional.region.is_single_trace
    assert regional.region.trace_indices == (2,)
    assert (regional.n_positions, regional.n_present, regional.n_missing) == (1, 1, 0)
    np.testing.assert_allclose(
        regional.spectrum.original, np.abs(np.fft.rfft(section.original[2])), rtol=1e-6
    )
    np.testing.assert_allclose(
        regional.spectrum.filtered, np.abs(np.fft.rfft(section.filtered[2])), rtol=1e-6
    )
    wide = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 5), DATASET, JOB
    )
    assert not wide.region.is_single_trace


def test_region_without_any_present_trace_is_not_valid_for_qc():
    section = make_section(sines([10] * 5), physical=[0, -1, -1, 3, 4])
    empty = SectionRegion(2, 3).resolve(section)  # two gaps, nothing else
    assert (empty.n_positions, empty.n_present, empty.n_missing) == (2, 0, 2)
    assert not empty.is_valid
    assert MIN_REGION_TRACES == 1
    one = SectionRegion(1, 3).resolve(section)  # one trace + two gaps
    assert one.is_valid and one.is_single_trace
    with pytest.raises(RegionTooSmallError, match="at least 1"):
        SeismicViewerService.regional_spectrum(
            section, SectionRegion(2, 3), DATASET, JOB
        )


# --- regional aggregate -------------------------------------------------------


def test_regional_curves_are_the_mean_of_per_trace_amplitude_spectra():
    section = make_section(sines([10, 20, 40, 60, 80], [0, 0.3, 1.1, 2.0, 0.5]))

    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion.from_boundaries(4, 2), DATASET, JOB
    )

    rows = [1, 2, 3]  # coordinates 2..4
    np.testing.assert_allclose(
        regional.spectrum.original,
        reference_mean_spectrum(section.original[rows]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        regional.spectrum.filtered,
        reference_mean_spectrum(section.filtered[rows]),
        rtol=1e-6,
    )
    assert regional.region.trace_indices == (1, 2, 3)
    assert regional.n_present == 3 and regional.n_missing == 0
    assert regional.dataset is DATASET and regional.job is JOB
    assert "Mean of per-trace amplitude spectra" in RegionalSpectrum.METHOD


def test_gaps_inside_the_region_never_enter_the_mean():
    traces = sines([10, 20, 30, 40, 50])
    section = make_section(traces, physical=[0, 1, -1, 3, -1])  # NaN rows at 2 and 4

    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 5), DATASET, JOB
    )

    assert np.isfinite(regional.spectrum.original).all()
    assert (regional.n_positions, regional.n_present, regional.n_missing) == (5, 3, 2)
    # denominator is the number of present traces (3), not positions (5)
    np.testing.assert_allclose(
        regional.spectrum.original,
        reference_mean_spectrum(section.original[[0, 1, 3]]),
        rtol=1e-6,
    )
    zeros_counted = reference_mean_spectrum(np.nan_to_num(section.original))
    assert not np.allclose(regional.spectrum.original, zeros_counted)


def test_regional_mean_is_not_the_spectrum_of_the_averaged_traces():
    # Same 20 Hz sine in anti-phase: the time-domain mean is ~0 everywhere,
    # but each trace carries a strong 20 Hz line that the QC must show.
    section = make_section(sines([20, 20], [0, np.pi]))

    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 2), DATASET, JOB
    )

    wrong = np.abs(np.fft.rfft(section.original.mean(axis=0)))
    peak = np.argmin(np.abs(regional.spectrum.frequencies_hz - 20))
    assert wrong[peak] < 1e-3 * regional.spectrum.original[peak]  # cancelled
    assert regional.spectrum.original[peak] > 100  # preserved per-trace energy
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(regional.spectrum.original, wrong, rtol=1e-3)


@pytest.mark.parametrize(
    "kind,upper,labels",
    [
        (FilterType.LOW_PASS, None, ["High cutoff"]),
        (FilterType.HIGH_PASS, None, ["Low cutoff"]),
        (FilterType.BAND_PASS, 50.0, ["Low cutoff", "High cutoff"]),
    ],
)
def test_axis_markers_and_single_zero_phase_response_follow_the_job(
    kind, upper, labels
):
    regional = make_regional(kind, 15.0, upper, n_traces=3)
    spectrum = regional.spectrum

    np.testing.assert_array_equal(spectrum.frequencies_hz, np.fft.rfftfreq(N, 0.004))
    assert spectrum.frequencies_hz[0] == 0
    assert spectrum.frequencies_hz[-1] <= spectrum.nyquist_hz == 125.0
    assert spectrum.filter_type is kind
    assert (spectrum.cutoff_hz, spectrum.upper_cutoff_hz) == (15.0, upper)
    assert [m.label for m in spectrum.cutoff_markers] == labels
    assert [m.frequency_hz for m in spectrum.cutoff_markers] == (
        [15.0, 50.0] if upper else [15.0]
    )
    btype = {
        FilterType.LOW_PASS: "lowpass",
        FilterType.HIGH_PASS: "highpass",
        FilterType.BAND_PASS: "bandpass",
    }[kind]
    sos = butter(4, [15, 50] if upper else 15, btype=btype, fs=250, output="sos")
    _, h = sosfreqz(sos, worN=spectrum.frequencies_hz, fs=250)
    # ONE response for the region -- |H|^2 of the job's SOS, independent of N
    np.testing.assert_allclose(spectrum.filter_response, np.abs(h) ** 2)
    assert spectrum.filter_response.shape == spectrum.frequencies_hz.shape
    assert spectrum.scale is SpectrumScale.LINEAR


def test_only_one_theoretical_response_is_computed(monkeypatch):
    import giecar_seismic.application.seismic_viewer as module

    calls = []
    real = module.sosfreqz

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "sosfreqz", counting)
    make_regional(n_traces=7)
    assert len(calls) == 1


# --- dB after linear aggregation ---------------------------------------------------


def test_db_is_applied_after_linear_aggregation_with_the_shared_rules():
    section = make_section(sines([20, 20, 40], [0, 0.5, 1.0]))
    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 3), DATASET, JOB
    )

    db = regional_spectrum_for_display(
        regional, SpectrumScale.DB, show_filter_response=True
    )

    linear = regional.spectrum
    peak = np.argmax(linear.original)
    expected = spectrum_for_display(linear, SpectrumScale.DB, show_filter_response=True)
    np.testing.assert_array_equal(db.spectrum.original, expected.original)
    np.testing.assert_array_equal(db.spectrum.filtered, expected.filtered)
    np.testing.assert_array_equal(db.spectrum.filter_response, expected.filter_response)
    assert db.spectrum.original[peak] == pytest.approx(0)  # aggregate peak = 0 dB
    assert db.spectrum.filtered[peak] == pytest.approx(-20)  # filtered = 0.1 x original
    assert db.spectrum.reference_magnitude == pytest.approx(linear.original[peak])
    assert db.spectrum.scale is SpectrumScale.DB and db.spectrum.show_filter_response
    np.testing.assert_allclose(
        db.spectrum.filter_response,
        20 * np.log10(np.maximum(linear.filter_response, 1e-6)),
    )
    # NOT the mean of per-trace dB spectra: at 40 Hz the aggregate is
    # (0 + 0 + A)/3 against a 2A/3 peak (about -6 dB); averaging per-trace
    # dB would mix two -120 dB floors with one 0 dB (about -80 dB).
    bin_40 = np.argmin(np.abs(linear.frequencies_hz - 40))
    assert db.spectrum.original[bin_40] == pytest.approx(-6.02, abs=0.1)
    per_trace_db = np.mean(
        [
            spectrum_for_display(
                replace(linear, original=np.abs(np.fft.rfft(section.original[i]))),
                SpectrumScale.DB,
            ).original
            for i in range(3)
        ],
        axis=0,
    )
    assert per_trace_db[bin_40] < -60
    # identity/metadata survive the transform
    assert db.region == regional.region and db.job is JOB


def test_display_toggles_never_mutate_or_recompute_the_raw_regional_spectrum(
    monkeypatch,
):
    regional = make_regional(n_traces=3)
    raw_original = regional.spectrum.original.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("display transform recomputed an FFT")

    monkeypatch.setattr(np.fft, "rfft", forbidden)
    db = regional_spectrum_for_display(regional, SpectrumScale.DB)
    back = regional_spectrum_for_display(regional, SpectrumScale.LINEAR)
    again = regional_spectrum_for_display(
        regional, SpectrumScale.DB, show_filter_response=True
    )

    assert back.spectrum.original is regional.spectrum.original
    np.testing.assert_array_equal(regional.spectrum.original, raw_original)
    np.testing.assert_array_equal(again.spectrum.original, db.spectrum.original)
    assert regional.spectrum.scale is SpectrumScale.LINEAR
    with pytest.raises(ValueError, match="linear"):
        regional_spectrum_for_display(db, SpectrumScale.LINEAR)  # never dB -> dB


@pytest.mark.parametrize("scale", [0.0, 1e-250])
def test_all_zero_or_tiny_region_stays_finite_at_the_db_floor(scale):
    section = make_section(sines([20, 20]) * scale)
    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 2), DATASET, JOB
    )
    with np.errstate(divide="raise", invalid="raise"):
        db = regional_spectrum_for_display(
            regional, SpectrumScale.DB, show_filter_response=True
        )
    assert np.isfinite(db.spectrum.original).all()
    assert np.isfinite(db.spectrum.filtered).all()
    assert np.isfinite(db.spectrum.filter_response).all()
    if scale == 0.0:
        np.testing.assert_array_equal(db.spectrum.original, -120)
        np.testing.assert_array_equal(db.spectrum.filtered, -120)
        assert db.spectrum.reference_magnitude == 1.0


def test_float32_zero_rows_convert_to_db_without_warnings():
    section = make_section(np.zeros((2, N)))
    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 2), DATASET, JOB
    )
    with np.errstate(divide="raise", invalid="raise"):
        db = regional_spectrum_for_display(regional, SpectrumScale.DB)
    np.testing.assert_array_equal(db.spectrum.original, -120)


# --- bounded, chunked aggregation --------------------------------------------------


@pytest.mark.parametrize(
    "n_traces",
    [
        REGION_SPECTRUM_CHUNK_TRACES - 1,
        REGION_SPECTRUM_CHUNK_TRACES,
        REGION_SPECTRUM_CHUNK_TRACES + 1,
        2 * REGION_SPECTRUM_CHUNK_TRACES + 5,  # partial final chunk
        3,
    ],
)
def test_chunked_aggregation_matches_the_straightforward_reference(n_traces):
    rng = np.random.default_rng(n_traces)
    traces = rng.standard_normal((n_traces, N))
    physical = np.arange(n_traces)
    physical[::7] = -1  # sprinkle gaps
    section = make_section(traces, physical=physical)
    present = np.flatnonzero(physical >= 0)
    assert len(present) >= 2

    regional = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, n_traces), DATASET, JOB
    )

    np.testing.assert_allclose(
        regional.spectrum.original,
        reference_mean_spectrum(section.original[present]),
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        regional.spectrum.filtered,
        reference_mean_spectrum(section.filtered[present]),
        rtol=1e-6,
        atol=1e-9,
    )
    assert regional.n_present == len(present)


def test_no_fft_call_ever_sees_more_rows_than_the_chunk_size(monkeypatch):
    n_traces = 3 * REGION_SPECTRUM_CHUNK_TRACES + 7
    section = make_section(np.random.default_rng(0).standard_normal((n_traces, N)))
    seen_rows: list[int] = []
    real = np.fft.rfft

    def recording(a, *args, **kwargs):
        seen_rows.append(1 if np.ndim(a) == 1 else np.shape(a)[0])
        return real(a, *args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft", recording)

    SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, n_traces), DATASET, JOB
    )

    assert seen_rows, "aggregation must go through np.fft.rfft"
    assert max(seen_rows) <= REGION_SPECTRUM_CHUNK_TRACES
    assert sum(seen_rows) == 2 * n_traces  # original + filtered, each trace once
    assert REGION_SPECTRUM_CHUNK_TRACES == 64


def test_chunk_size_is_configurable_and_validated():
    section = make_section(sines([10] * 5))
    small = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 5), DATASET, JOB, chunk_traces=2
    )
    big = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 5), DATASET, JOB, chunk_traces=1000
    )
    np.testing.assert_allclose(
        small.spectrum.original, big.spectrum.original, rtol=1e-6
    )
    with pytest.raises(ValueError, match="chunk_traces"):
        SeismicViewerService.regional_spectrum(
            section, SectionRegion(1, 5), DATASET, JOB, chunk_traces=0
        )


# --- single-trace waveform (amplitude x time) -----------------------------------


def test_single_trace_region_carries_its_waveform_and_wider_regions_do_not():
    section = make_section(sines([10, 20, 30], [0, 1, 2]))

    single = SeismicViewerService.regional_spectrum(
        section, SectionRegion(2, 2), DATASET, JOB
    )
    waveform = single.waveform
    assert isinstance(waveform, TraceWaveform)
    assert waveform.trace_index == 1 and waveform.coordinate == 2
    np.testing.assert_array_equal(waveform.original, section.original[1])
    np.testing.assert_array_equal(waveform.filtered, section.filtered[1])
    np.testing.assert_allclose(waveform.time_ms, np.arange(N) * 4.0)
    assert waveform.amplitude_scale == pytest.approx(
        np.nanmax(np.abs(np.concatenate([waveform.original, waveform.filtered])))
    )
    # a copy of that one row, not a view of the whole section
    assert not np.shares_memory(waveform.original, section.original)

    wide = SeismicViewerService.regional_spectrum(
        section, SectionRegion(1, 3), DATASET, JOB
    )
    assert wide.waveform is None  # never a time-domain mean of the region
    assert regional_spectrum_for_display(single, SpectrumScale.DB).waveform is waveform
