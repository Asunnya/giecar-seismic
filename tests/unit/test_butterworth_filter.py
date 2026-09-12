import numpy as np

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter


def test_output_shape_matches_input_shape():
    chunk = np.zeros((4, 850), dtype=np.float32)

    filtered = apply_lowpass_filter(chunk, cutoff_hz=30.0, order=4, sample_rate_ms=4.0)

    assert filtered.shape == chunk.shape


def _sine_trace(
    frequency_hz: float, sample_rate_ms: float, n_samples: int
) -> np.ndarray:
    sampling_frequency_hz = 1000 / sample_rate_ms
    t = np.arange(n_samples) / sampling_frequency_hz
    return np.sin(2 * np.pi * frequency_hz * t)


def _middle_amplitude_ratio(original: np.ndarray, filtered: np.ndarray) -> float:
    # Zero-phase filtering (filtfilt) still has edge transients near the
    # boundaries; compare amplitude only over the trace's middle portion.
    n = len(original)
    edge = n // 4
    middle = slice(edge, n - edge)
    return float(np.abs(filtered[middle]).max() / np.abs(original[middle]).max())


def test_frequency_well_below_cutoff_passes_through():
    sample_rate_ms = 4.0  # fs=250 Hz, Nyquist=125 Hz
    trace = _sine_trace(frequency_hz=5.0, sample_rate_ms=sample_rate_ms, n_samples=2000)
    chunk = trace[np.newaxis, :]

    filtered = apply_lowpass_filter(
        chunk, cutoff_hz=30.0, order=4, sample_rate_ms=sample_rate_ms
    )

    assert _middle_amplitude_ratio(trace, filtered[0]) > 0.95


def test_frequency_well_above_cutoff_is_attenuated():
    sample_rate_ms = 4.0  # fs=250 Hz, Nyquist=125 Hz
    trace = _sine_trace(
        frequency_hz=100.0, sample_rate_ms=sample_rate_ms, n_samples=2000
    )
    chunk = trace[np.newaxis, :]

    filtered = apply_lowpass_filter(
        chunk, cutoff_hz=30.0, order=4, sample_rate_ms=sample_rate_ms
    )

    assert _middle_amplitude_ratio(trace, filtered[0]) < 0.1


def test_each_trace_in_a_chunk_is_filtered_independently():
    sample_rate_ms = 4.0
    low_freq_trace = _sine_trace(
        frequency_hz=5.0, sample_rate_ms=sample_rate_ms, n_samples=2000
    )
    high_freq_trace = _sine_trace(
        frequency_hz=100.0, sample_rate_ms=sample_rate_ms, n_samples=2000
    )
    chunk = np.stack([low_freq_trace, high_freq_trace])

    filtered = apply_lowpass_filter(
        chunk, cutoff_hz=30.0, order=4, sample_rate_ms=sample_rate_ms
    )

    assert _middle_amplitude_ratio(low_freq_trace, filtered[0]) > 0.95
    assert _middle_amplitude_ratio(high_freq_trace, filtered[1]) < 0.1
