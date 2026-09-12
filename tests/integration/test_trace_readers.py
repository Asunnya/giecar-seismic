"""Selective readers used by the viewer: the SEG-Y reader's read_traces(),
the segyio header batch iterator, and the HDF5 output reader. Every
assertion is derived from the fixture, never from survey-specific numbers.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import segyio

from giecar_seismic.infrastructure.segy.reader import (
    SegyTraceReader,
    iter_trace_header_batches,
)
from giecar_seismic.infrastructure.storage.hdf5_reader import Hdf5TraceReader
from giecar_seismic.infrastructure.storage.hdf5_writer import Hdf5TraceWriter

HEADERS = [(10, 1), (10, 2), (10, 3), (11, 1), (11, 2), (12, 1), (12, 2), (12, 3)]
N_SAMPLES = 16


@pytest.fixture
def segy_path(tmp_path: Path) -> tuple[Path, np.ndarray]:
    path = tmp_path / "s.segy"
    amplitudes = np.arange(len(HEADERS) * N_SAMPLES, dtype=np.float32).reshape(
        len(HEADERS), N_SAMPLES
    )
    spec = segyio.spec()
    spec.samples = list(range(N_SAMPLES))
    spec.tracecount = len(HEADERS)
    spec.format = 5
    with segyio.create(str(path), spec) as segy:
        for i, (il, xl) in enumerate(HEADERS):
            segy.trace[i] = amplitudes[i]
            segy.header[i][segyio.TraceField.INLINE_3D] = il
            segy.header[i][segyio.TraceField.CROSSLINE_3D] = xl
        segy.bin[segyio.BinField.Interval] = 4000
    return path, amplitudes


def test_header_batches_stream_the_geometry_in_bounded_batches(segy_path):
    path, _ = segy_path

    batches = list(iter_trace_header_batches(str(path), batch_size=3))

    assert [len(b) for b in batches] == [3, 3, 2]
    flat = [(g.trace_index, g.inline, g.crossline) for b in batches for g in b]
    assert flat == [(i, il, xl) for i, (il, xl) in enumerate(HEADERS)]


def test_segy_read_traces_returns_only_the_requested_physical_indices(segy_path):
    path, amplitudes = segy_path
    with SegyTraceReader(path) as reader:
        out = reader.read_traces([6, 1, 3])

    assert out.shape == (3, N_SAMPLES)  # proportional to the request
    np.testing.assert_array_equal(out, amplitudes[[6, 1, 3]])  # request order kept


def test_segy_read_traces_rejects_out_of_range_indices(segy_path):
    path, _ = segy_path
    with SegyTraceReader(path) as reader:
        with pytest.raises(ValueError):
            reader.read_traces([0, len(HEADERS)])
        with pytest.raises(ValueError):
            reader.read_traces([-1])


def test_hdf5_reader_reads_only_the_requested_indices_and_closes(tmp_path: Path):
    path = tmp_path / "out.h5"
    data = np.arange(8 * N_SAMPLES, dtype=np.float32).reshape(8, N_SAMPLES) * 0.5
    writer = Hdf5TraceWriter(path, expected_trace_count=8, n_samples=N_SAMPLES)
    writer.write_chunk(0, data)
    writer.finalize()
    writer.close()

    with Hdf5TraceReader(path) as reader:
        assert reader.trace_count == 8
        assert reader.n_samples == N_SAMPLES
        assert reader.complete is True
        out = reader.read_traces([7, 0, 2])
    assert out.shape == (3, N_SAMPLES)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, data[[7, 0, 2]])

    # the handle really is released: the file can be reopened for writing
    with h5py.File(path, "a"):
        pass


def test_hdf5_reader_rejects_out_of_range_indices(tmp_path: Path):
    path = tmp_path / "out.h5"
    writer = Hdf5TraceWriter(path, expected_trace_count=2, n_samples=N_SAMPLES)
    writer.write_chunk(0, np.zeros((2, N_SAMPLES), dtype=np.float32))
    writer.close()

    with Hdf5TraceReader(path) as reader, pytest.raises(ValueError):
        reader.read_traces([2])
