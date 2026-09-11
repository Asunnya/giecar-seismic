import numpy as np
from scipy.signal import butter, sosfiltfilt


def apply_lowpass_filter(
    chunk: np.ndarray, cutoff_hz: float, order: int, sample_rate_ms: float
) -> np.ndarray:
    sampling_frequency_hz = 1000 / sample_rate_ms
    sos = butter(order, cutoff_hz, btype="low", fs=sampling_frequency_hz, output="sos")
    return sosfiltfilt(sos, chunk, axis=-1)
