from concurrent.futures import Future

import numpy as np
import pytest

from giecar_seismic.application.parallel_filter import (
    filter_chunk_with_executor,
    partition_trace_ranges,
)


class ImmediateExecutor:
    def __init__(self) -> None:
        self.partitions: list[np.ndarray] = []

    def submit(self, fn, *args):
        partition = args[1]
        self.partitions.append(partition)
        future: Future[np.ndarray] = Future()
        future.set_result(fn(*args))
        return future


@pytest.mark.parametrize("workers", [0, -1, -8])
def test_partition_trace_ranges_rejects_non_positive_workers(workers):
    with pytest.raises(ValueError, match="parallel_workers must be at least 1"):
        partition_trace_ranges(5, workers)


def test_chunk_smaller_than_worker_count_has_no_empty_partitions():
    assert partition_trace_ranges(3, 8) == ((0, 1), (1, 2), (2, 3))


def test_non_divisible_chunk_is_partitioned_contiguously_in_order():
    ranges = partition_trace_ranges(10, 4)
    assert ranges == ((0, 3), (3, 6), (6, 8), (8, 10))
    assert [index for start, stop in ranges for index in range(start, stop)] == list(
        range(10)
    )


def test_executor_receives_contiguous_partitions_and_result_keeps_trace_order():
    chunk = np.arange(7 * 24, dtype=np.float64).reshape(7, 24)
    # Identity SOS: b0=1, remaining coefficients zero except a0=1.
    sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    executor = ImmediateExecutor()

    result = filter_chunk_with_executor(chunk, sos, executor, parallel_workers=3)

    np.testing.assert_allclose(result, chunk)
    assert [part[:, 0].tolist() for part in executor.partitions] == [
        [0.0, 24.0, 48.0],
        [72.0, 96.0],
        [120.0, 144.0],
    ]
    assert result.shape == chunk.shape
    assert result.dtype == chunk.dtype
