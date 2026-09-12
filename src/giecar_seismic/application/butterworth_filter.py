from typing import Literal

import numpy as np
from scipy.signal import butter, sosfiltfilt

from giecar_seismic.domain.job import FilterType


def butterworth_sos(
    cutoff_hz: float,
    order: int,
    sample_rate_ms: float,
    *,
    filter_type: FilterType = FilterType.LOW_PASS,
    upper_cutoff_hz: float | None = None,
) -> np.ndarray:
    """Shared design for processing and theoretical frequency response.

    `order` is the prototype order passed to SciPy. A band-pass design
    has transfer-function order 2 * order; forward/backward sosfiltfilt
    then squares its magnitude response (zero phase), for every type.
    """
    types: dict[FilterType, Literal["lowpass", "highpass", "bandpass"]] = {
        FilterType.LOW_PASS: "lowpass",
        FilterType.HIGH_PASS: "highpass",
        FilterType.BAND_PASS: "bandpass",
    }
    btype = types[filter_type]
    frequencies: float | tuple[float, float] = cutoff_hz
    if filter_type is FilterType.BAND_PASS:
        if upper_cutoff_hz is None:
            raise ValueError("upper_cutoff_hz is required for BAND_PASS")
        frequencies = (cutoff_hz, upper_cutoff_hz)
    elif upper_cutoff_hz is not None:
        raise ValueError("upper_cutoff_hz must be None for LOW_PASS and HIGH_PASS")
    return butter(
        order, frequencies, btype=btype, fs=1000 / sample_rate_ms, output="sos"
    )


def apply_butterworth_filter(
    chunk: np.ndarray,
    cutoff_hz: float,
    order: int,
    sample_rate_ms: float,
    *,
    filter_type: FilterType = FilterType.LOW_PASS,
    upper_cutoff_hz: float | None = None,
) -> np.ndarray:
    sos = butterworth_sos(
        cutoff_hz,
        order,
        sample_rate_ms,
        filter_type=filter_type,
        upper_cutoff_hz=upper_cutoff_hz,
    )
    return sosfiltfilt(sos, chunk, axis=-1)


def apply_lowpass_filter(
    chunk: np.ndarray, cutoff_hz: float, order: int, sample_rate_ms: float
) -> np.ndarray:
    """Backward-compatible low-pass entry point."""
    return apply_butterworth_filter(chunk, cutoff_hz, order, sample_rate_ms)
