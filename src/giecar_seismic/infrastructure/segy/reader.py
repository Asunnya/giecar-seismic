from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import segyio

from giecar_seismic.domain.dataset import SeismicDataset


class SegyTraceReader:
    """Streaming trace access over a SEG-Y file via segyio.

    Uses ignore_geometry=True: production surveys may have an irregular
    inline/crossline footprint (see notebooks/01_inspect_segy.ipynb) that
    breaks segyio's strict geometry inference. Traces are addressed by
    physical sequential index, not inline/crossline coordinates.
    """

    def __init__(self, source_path: str | Path):
        self._segy = segyio.open(source_path, mode="r", ignore_geometry=True)
        self.trace_count: int = self._segy.tracecount
        self._sample_count = len(self._segy.samples)

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        if start < 0:
            raise ValueError("start must be non-negative")
        if stop <= start:
            raise ValueError("stop must be greater than start")
        if stop > self.trace_count:
            raise ValueError("stop exceeds trace count")

        dtype = np.asarray(self._segy.trace[start]).dtype
        chunk = np.empty((stop - start, self._sample_count), dtype=dtype)
        for row_index, trace_index in enumerate(range(start, stop)):
            chunk[row_index] = self._segy.trace[trace_index]
        return chunk

    def close(self) -> None:
        self._segy.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_dataset_reader(dataset: SeismicDataset) -> SegyTraceReader:
    """ReaderFactory for FilterJobService: re-opens the dataset's SEG-Y for
    streaming. The service cross-checks reader.trace_count against
    dataset.n_traces before processing.
    """
    return SegyTraceReader(dataset.source_path)
