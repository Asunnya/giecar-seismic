import json

import numpy as np
import pytest

from benchmarks.memory_benchmark import (
    BenchmarkResult,
    aggregate_medians,
    iter_trace_ranges,
    peak_rss_to_mb,
    stream_filter_to_hdf5,
    validate_request,
)


@pytest.mark.parametrize("strategy", ["fast", "all-at-once", ""])
def test_invalid_strategy_is_rejected(strategy):
    with pytest.raises(ValueError, match="strategy"):
        validate_request(strategy, 10, 4, file_trace_count=20)


@pytest.mark.parametrize("trace_count", [0, -1])
def test_non_positive_trace_count_is_rejected(trace_count):
    with pytest.raises(ValueError, match="trace_count must be positive"):
        validate_request("streaming", trace_count, 4, file_trace_count=20)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        validate_request("streaming", 10, chunk_size, file_trace_count=20)


def test_trace_count_larger_than_file_is_rejected():
    with pytest.raises(ValueError, match="exceeds SEG-Y physical trace count"):
        validate_request("naive", 21, 4, file_trace_count=20)


def test_ranges_cover_prefix_once_and_keep_smaller_last_chunk():
    assert list(iter_trace_ranges(10, 4)) == [(0, 4), (4, 8), (8, 10)]


class RecordingReader:
    trace_count = 10
    sample_count = 8

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        self.calls.append((start, stop))
        rows = np.arange(start, stop, dtype=np.float32)[:, None]
        return np.repeat(rows, self.sample_count, axis=1)


def test_streaming_reads_every_trace_once_and_writes_last_partial_chunk(
    tmp_path, monkeypatch
):
    reader = RecordingReader()
    monkeypatch.setattr(
        "benchmarks.memory_benchmark.apply_lowpass_filter",
        lambda chunk, cutoff, order, sample_rate: chunk + 1,
    )

    stream_filter_to_hdf5(
        reader,
        tmp_path / "streaming.h5",
        trace_count=10,
        chunk_size=4,
        cutoff_hz=30.0,
        order=4,
        sample_rate_ms=4.0,
        output_dtype=np.dtype("float32"),
    )

    assert reader.calls == [(0, 4), (4, 8), (8, 10)]
    import h5py

    with h5py.File(tmp_path / "streaming.h5", "r") as output:
        np.testing.assert_array_equal(
            output["traces"][:, 0], np.arange(1, 11, dtype=np.float32)
        )


def test_peak_rss_conversion_is_platform_specific():
    assert peak_rss_to_mb(2048, platform="linux") == 2.0
    assert peak_rss_to_mb(2 * 1024**2, platform="darwin") == 2.0


def test_result_json_contains_expected_fields():
    result = BenchmarkResult(
        strategy="streaming",
        trace_count=10,
        sample_count=8,
        input_size_mb=0.01,
        chunk_size=4,
        repetition=2,
        peak_rss_mb=123.0,
        elapsed_seconds=0.5,
    )
    assert set(json.loads(result.to_json())) == {
        "strategy",
        "trace_count",
        "sample_count",
        "input_size_mb",
        "chunk_size",
        "repetition",
        "peak_rss_mb",
        "elapsed_seconds",
    }


def test_aggregation_uses_median_for_each_strategy_and_trace_count():
    records = [
        {"strategy": "naive", "trace_count": 10, "peak_rss_mb": value,
         "additional_peak_mb": value - 10, "elapsed_seconds": value / 10}
        for value in (30.0, 10.0, 20.0)
    ]
    aggregated = aggregate_medians(records)
    assert aggregated == [
        {
            "strategy": "naive",
            "trace_count": 10,
            "median_peak_rss_mb": 20.0,
            "median_additional_peak_mb": 10.0,
            "median_elapsed_seconds": 2.0,
        }
    ]


def test_json_result_contains_analysis_fields():
    result = BenchmarkResult(
        strategy="streaming",
        trace_count=10,
        sample_count=8,
        input_size_mb=0.000305,
        chunk_size=4,
        repetition=2,
        peak_rss_mb=120.5,
        elapsed_seconds=0.25,
    )
    payload = json.loads(result.to_json())
    assert set(payload) == {
        "strategy",
        "trace_count",
        "sample_count",
        "input_size_mb",
        "chunk_size",
        "repetition",
        "peak_rss_mb",
        "elapsed_seconds",
    }
