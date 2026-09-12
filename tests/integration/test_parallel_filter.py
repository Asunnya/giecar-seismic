from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest

from giecar_seismic.application.butterworth_filter import (
    apply_butterworth_filter,
    butterworth_sos,
)
from giecar_seismic.application.parallel_filter import filter_chunk_with_executor
from giecar_seismic.domain.job import FilterType


@pytest.mark.parametrize(
    ("trace_count", "sample_count", "workers"), [(2, 192, 4), (7, 257, 3)]
)
def test_spawn_process_pool_matches_sequential_filter(
    trace_count, sample_count, workers
):
    rng = np.random.default_rng(20260912)
    chunk = rng.standard_normal((trace_count, sample_count)).astype(np.float32)

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        for filter_type, cutoff_hz, upper_cutoff_hz in (
            (FilterType.LOW_PASS, 35.0, None),
            (FilterType.HIGH_PASS, 12.0, None),
            (FilterType.BAND_PASS, 12.0, 35.0),
        ):
            kwargs = {
                "filter_type": filter_type,
                "upper_cutoff_hz": upper_cutoff_hz,
            }
            expected = apply_butterworth_filter(chunk, cutoff_hz, 4, 4.0, **kwargs)
            sos = butterworth_sos(cutoff_hz, 4, 4.0, **kwargs)
            actual = filter_chunk_with_executor(
                chunk, sos, executor, parallel_workers=workers
            )

            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype


def test_spawn_process_failure_is_propagated_to_coordinator():
    chunk = np.zeros((2, 4), dtype=np.float32)
    sos = butterworth_sos(35.0, 4, 4.0)

    with (
        ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as executor,
        pytest.raises(ValueError, match="length of the input vector"),
    ):
        filter_chunk_with_executor(chunk, sos, executor, parallel_workers=2)
