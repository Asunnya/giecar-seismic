from concurrent.futures import Executor

import numpy as np
from scipy.signal import sosfiltfilt


def partition_trace_ranges(
    trace_count: int, parallel_workers: int
) -> tuple[tuple[int, int], ...]:
    """Split a chunk into at most ``parallel_workers`` contiguous ranges."""
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or parallel_workers < 1
    ):
        raise ValueError("parallel_workers must be at least 1")
    if trace_count < 0:
        raise ValueError("trace_count must not be negative")
    partition_count = min(trace_count, parallel_workers)
    if partition_count == 0:
        return ()

    base_size, remainder = divmod(trace_count, partition_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(partition_count):
        stop = start + base_size + (1 if index < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def filter_sos_partition(sos: np.ndarray, partition: np.ndarray) -> np.ndarray:
    """Pure, spawn-serializable CPU task for one group of traces."""
    return sosfiltfilt(sos, partition, axis=-1)


def filter_chunk_with_executor(
    chunk: np.ndarray,
    sos: np.ndarray,
    executor: Executor,
    *,
    parallel_workers: int,
) -> np.ndarray:
    """Filter contiguous partitions and restore their physical input order."""
    ranges = partition_trace_ranges(chunk.shape[0], parallel_workers)
    if not ranges:
        return chunk.copy()
    futures = [
        executor.submit(filter_sos_partition, sos, chunk[start:stop])
        for start, stop in ranges
    ]
    # Futures are consumed in submission/range order. Completion order cannot
    # change the physical trace order expected by HDF5, resume, and geometry.
    return np.concatenate([future.result() for future in futures], axis=0)
