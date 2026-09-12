from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import segyio

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import TraceGeometry


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

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def sample_rate_ms(self) -> float:
        return float(segyio.tools.dt(self._segy)) / 1000

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

    def read_traces(self, trace_indices: Sequence[int]) -> np.ndarray:
        """Read an arbitrary, small collection of physical traces, in the
        order requested. Memory is proportional to len(trace_indices) --
        the viewer uses this for one section at a time.
        """
        for index in trace_indices:
            if not 0 <= index < self.trace_count:
                raise ValueError(f"trace index {index} outside [0, {self.trace_count})")
        out = np.empty((len(trace_indices), self._sample_count), dtype=np.float32)
        for row, index in enumerate(trace_indices):
            out[row] = self._segy.trace[index]
        return out

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


def _iter_inline_crossline_batches(
    segy, batch_size: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield bounded header arrays from an already-open physical trace stream.

    This is the single slicing primitive shared by metadata import and geometry
    indexing. It retains no prior batch and never accesses trace amplitudes.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total = segy.tracecount
    inlines = segy.attributes(segyio.TraceField.INLINE_3D)
    crosslines = segy.attributes(segyio.TraceField.CROSSLINE_3D)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        yield (
            start,
            np.asarray(inlines[start:stop]),
            np.asarray(crosslines[start:stop]),
        )


def iter_trace_header_batches(
    source_path: str, batch_size: int
) -> Iterator[list[TraceGeometry]]:
    """TraceHeaderBatchReader for BuildGeometryIndexUseCase.

    Reads INLINE_3D/CROSSLINE_3D for `batch_size` traces at a time via
    segyio attribute slices. Each batch is a small array, and nothing
    accumulates across batches.
    """
    with segyio.open(source_path, mode="r", ignore_geometry=True) as segy:
        for start, batch_inlines, batch_crosslines in _iter_inline_crossline_batches(
            segy, batch_size
        ):
            yield [
                TraceGeometry(
                    trace_index=start + offset,
                    inline=int(batch_inlines[offset]),
                    crossline=int(batch_crosslines[offset]),
                )
                for offset in range(len(batch_inlines))
            ]
