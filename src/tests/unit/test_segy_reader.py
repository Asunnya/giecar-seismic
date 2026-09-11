from pathlib import Path

import numpy as np
import pytest
import segyio

from giecar_seismic.infrastructure.segy.reader import SegyTraceReader

N_TRACES = 5
N_SAMPLES = 10
SAMPLE_INTERVAL_US = 4000


@pytest.fixture
def tiny_segy_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.segy"

    spec = segyio.spec()
    spec.samples = list(range(N_SAMPLES))
    spec.tracecount = N_TRACES
    spec.format = 5  # IEEE float32

    with segyio.create(str(path), spec) as segy:
        for i in range(N_TRACES):
            segy.trace[i] = np.arange(N_SAMPLES, dtype=np.float32) + i
        segy.bin[segyio.BinField.Interval] = SAMPLE_INTERVAL_US

    return path


def test_trace_count_matches_file(tiny_segy_path):
    reader = SegyTraceReader(tiny_segy_path)
    try:
        assert reader.trace_count == N_TRACES
    finally:
        reader.close()


def test_read_chunk_returns_expected_traces(tiny_segy_path):
    reader = SegyTraceReader(tiny_segy_path)
    try:
        chunk = reader.read_chunk(1, 3)

        assert chunk.shape == (2, N_SAMPLES)
        np.testing.assert_array_equal(chunk[0], np.arange(N_SAMPLES, dtype=np.float32) + 1)
        np.testing.assert_array_equal(chunk[1], np.arange(N_SAMPLES, dtype=np.float32) + 2)
    finally:
        reader.close()


def test_read_chunk_dtype_matches_file_dtype(tiny_segy_path):
    reader = SegyTraceReader(tiny_segy_path)
    try:
        chunk = reader.read_chunk(0, 1)
        assert chunk.dtype == np.float32
    finally:
        reader.close()


def test_context_manager_closes_the_underlying_file(tiny_segy_path):
    with SegyTraceReader(tiny_segy_path) as reader:
        assert reader.trace_count == N_TRACES

    with pytest.raises(OSError):
        reader.read_chunk(0, 1)


@pytest.mark.parametrize(
    "start,stop",
    [(-1, 2), (2, 2), (3, 1), (0, N_TRACES + 1)],
)
def test_read_chunk_rejects_invalid_bounds(tiny_segy_path, start, stop):
    reader = SegyTraceReader(tiny_segy_path)
    try:
        with pytest.raises(ValueError):
            reader.read_chunk(start, stop)
    finally:
        reader.close()
