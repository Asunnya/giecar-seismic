from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import h5py
import numpy as np

from giecar_seismic.domain.job import Job
from giecar_seismic.infrastructure.storage.hdf5_writer import TRACES_DATASET_NAME


class Hdf5TraceReader:
    """Selective, read-only access to a filter job's HDF5 output.

    Separate from Hdf5TraceWriter on purpose (write vs. read lifecycles).
    read_traces() pulls only the requested physical indices -- the file
    keeps the SEG-Y's physical order, so the same index addresses the
    original and the filtered trace. Nothing here ever reads the whole
    `traces` dataset.
    """

    def __init__(self, output_path: str | Path) -> None:
        self._file = h5py.File(output_path, mode="r")
        self._traces = self._file[TRACES_DATASET_NAME]
        self.trace_count: int = int(self._traces.shape[0])
        self.n_samples: int = int(self._traces.shape[1])
        self.complete: bool = bool(self._file.attrs.get("complete", False))

    def read_traces(self, trace_indices: Sequence[int]) -> np.ndarray:
        for index in trace_indices:
            if not 0 <= index < self.trace_count:
                raise ValueError(f"trace index {index} outside [0, {self.trace_count})")
        out = np.empty((len(trace_indices), self.n_samples), dtype=np.float32)
        for row, index in enumerate(trace_indices):
            out[row] = self._traces[index]
        return out

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_job_output_reader(job: Job) -> Hdf5TraceReader:
    """FilteredReaderFactory for SeismicViewerService."""
    if job.output_path is None:
        raise ValueError(f"job {job.id} has no output to read")
    return Hdf5TraceReader(job.output_path)
