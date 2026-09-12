from collections.abc import Callable, Iterator
from typing import Protocol

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import TraceGeometry

# Port: yields the survey's trace headers as successive bounded batches of
# TraceGeometry, in physical order. Memory is O(batch_size), never
# O(n_traces). infrastructure.segy provides the segyio implementation.
TraceHeaderBatchReader = Callable[[str, int], Iterator[list[TraceGeometry]]]

DEFAULT_HEADER_BATCH_SIZE = 4096


class GeometryIndexWriter(Protocol):
    """Write side of the geometry index (queries live on GeometryRepository
    in seismic_viewer.py); SqlAlchemyGeometryRepository implements both.
    """

    def count(self, dataset_id: int) -> int: ...
    def add_batch(self, dataset_id: int, batch: list[TraceGeometry]) -> None: ...
    def delete_for_dataset(self, dataset_id: int) -> None: ...


class IncompleteGeometryIndexError(Exception):
    """The number of headers read from the SEG-Y does not match the
    dataset's physical trace count, so the index cannot be trusted.
    """


class BuildGeometryIndexUseCase:
    """Build the persistent (dataset, inline, crossline) -> trace index map.

    Completeness is proven by `count(dataset_id) == dataset.n_traces`; a
    complete index is never rebuilt. Anything else (missing or partial,
    e.g. an interrupted earlier build) is dropped and rebuilt in full --
    deliberately no resume logic, so the index can never mix stale and
    fresh rows. Headers are streamed in batches and persisted batch by
    batch; nothing here ever holds the whole survey's geometry.

    Framework-free (no Qt, no SQLAlchemy). Returns True if a build ran.
    """

    def __init__(
        self,
        read_header_batches: TraceHeaderBatchReader,
        geometry: GeometryIndexWriter,
        batch_size: int = DEFAULT_HEADER_BATCH_SIZE,
    ) -> None:
        self._read_header_batches = read_header_batches
        self._geometry = geometry
        self._batch_size = batch_size

    def __call__(self, dataset: SeismicDataset) -> bool:
        if dataset.id is None:
            raise ValueError("cannot index a dataset that has no id (never persisted)")
        if self._geometry.count(dataset.id) == dataset.n_traces:
            return False

        self._geometry.delete_for_dataset(dataset.id)
        written = 0
        for batch in self._read_header_batches(dataset.source_path, self._batch_size):
            self._geometry.add_batch(dataset.id, batch)
            written += len(batch)

        if written != dataset.n_traces:
            # leave it clearly incomplete (count != n_traces) rather than
            # passing as valid; the next attempt will rebuild from scratch.
            raise IncompleteGeometryIndexError(
                f"read {written} trace headers but dataset {dataset.id} has "
                f"{dataset.n_traces} traces"
            )
        return True
