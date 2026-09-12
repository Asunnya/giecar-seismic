from pathlib import Path
from types import TracebackType
from typing import Self

import h5py
import numpy as np
import numpy.typing as npt

TRACES_DATASET_NAME = "traces"

DEFAULT_STORAGE_CHUNK_TRACES = 256

# float32 (not float64): the SEG-Y sources this pipeline reads store
# amplitudes as float32, and SciPy's filtfilt/lfilter may return a float64
# temporary for a single chunk -- persisting in float32 avoids doubling the
# output's size on disk without losing precision the source never had.
# Conversion happens per chunk, never over the full volume.
DEFAULT_DTYPE: npt.DTypeLike = np.float32


class Hdf5WriterError(Exception):
    """Raised for any invalid use of Hdf5TraceWriter: invalid constructor
    configuration, writing after finalize()/close(), a malformed chunk,
    an out-of-order or overflowing write, or finalizing before all
    expected traces were written.
    """


class Hdf5TraceWriter:
    """Streaming, chunked HDF5 writer for filtered seismic traces.

    Persists incrementally to a single `traces` dataset of shape
    `(written_trace_count, n_samples)`, resized only as far as each
    write_chunk() call actually advances it -- never pre-materialized to
    `expected_trace_count`. Peak memory is bounded by one chunk, never by
    the full survey.

    Lifecycle mirrors the project's TraceWriter contract: finalize()
    marks the output as semantically complete and valid (only once every
    expected trace has been written); close() only releases the HDF5
    handle and never promotes a partial output to complete. A file that
    was only close()d, without finalize(), remains explicitly partial
    (`complete=False` in its attrs) and is still fully reopenable for the
    traces it does hold.
    """

    def __init__(
        self,
        output_path: str | Path,
        expected_trace_count: int,
        n_samples: int,
        chunk_size: int = DEFAULT_STORAGE_CHUNK_TRACES,
        dtype: npt.DTypeLike = DEFAULT_DTYPE,
    ) -> None:
        # Validated before anything touches HDF5: an invalid shape/chunk
        # dimension would otherwise surface as an opaque h5py/HDF5 error,
        # and would already have created an (empty, broken) file on disk.
        if expected_trace_count <= 0:
            raise Hdf5WriterError(
                f"expected_trace_count must be positive, got {expected_trace_count}"
            )
        if n_samples <= 0:
            raise Hdf5WriterError(f"n_samples must be positive, got {n_samples}")
        if chunk_size <= 0:
            raise Hdf5WriterError(f"chunk_size must be positive, got {chunk_size}")

        self._expected_trace_count = expected_trace_count
        self._n_samples = n_samples
        self._dtype = np.dtype(dtype)
        self._written = 0
        self._closed = False
        self._finalized = False

        self._file = h5py.File(output_path, mode="w")
        storage_chunk_traces = min(chunk_size, expected_trace_count)
        self._traces = self._file.create_dataset(
            TRACES_DATASET_NAME,
            shape=(0, n_samples),
            maxshape=(expected_trace_count, n_samples),
            dtype=self._dtype,
            chunks=(storage_chunk_traces, n_samples),
        )
        self._file.attrs["expected_trace_count"] = expected_trace_count
        self._file.attrs["n_samples"] = n_samples
        self._file.attrs["written_trace_count"] = 0
        self._file.attrs["complete"] = False

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        if self._closed:
            raise Hdf5WriterError("cannot write to a closed writer")
        if self._finalized:
            raise Hdf5WriterError("cannot write to an already-finalized writer")
        if chunk.ndim != 2:
            raise Hdf5WriterError(f"chunk must be 2D, got {chunk.ndim}D")
        if chunk.shape[1] != self._n_samples:
            raise Hdf5WriterError(
                f"chunk has {chunk.shape[1]} samples per trace, expected "
                f"{self._n_samples}"
            )
        if start != self._written:
            raise Hdf5WriterError(
                f"expected start={self._written} (the next trace after "
                f"{self._written} already written), got start={start} -- "
                "out-of-order, gapped and overlapping writes are not "
                "supported"
            )

        stop = start + chunk.shape[0]
        if stop > self._expected_trace_count:
            raise Hdf5WriterError(
                f"writing traces [{start}:{stop}) would exceed the "
                f"expected trace count of {self._expected_trace_count}"
            )

        self._traces.resize(stop, axis=0)
        self._traces[start:stop] = chunk.astype(self._dtype, copy=False)
        self._written = stop
        self._file.attrs["written_trace_count"] = self._written

    def finalize(self) -> None:
        if self._closed:
            raise Hdf5WriterError("cannot finalize a closed writer")
        if self._finalized:
            raise Hdf5WriterError("writer is already finalized")
        if self._written != self._expected_trace_count:
            raise Hdf5WriterError(
                f"cannot finalize: only {self._written} of "
                f"{self._expected_trace_count} expected traces have been "
                "written"
            )

        # Flush before marking complete, not after: if flush() raises,
        # execution never reaches the line that claims the output is
        # complete. This guarantees -- by construction, not by rolling
        # back on failure -- that a finalize() call which raises can
        # never leave complete=True, and that a subsequent close() can
        # never promote a failed finalize() to complete either.
        self._file.flush()
        self._file.attrs["complete"] = True
        self._finalized = True

    def close(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
