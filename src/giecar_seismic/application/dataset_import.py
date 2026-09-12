from collections.abc import Callable
from typing import Protocol

from giecar_seismic.domain.dataset import SeismicDataset

# Port: reads SEG-Y metadata (headers only) and returns an *unpersisted*
# SeismicDataset (id=None). infrastructure.segy.import_segy_dataset is the
# real implementation; tests supply plain functions.
DatasetMetadataReader = Callable[[str, str], SeismicDataset]


class DatasetWriteRepository(Protocol):
    """Write-side port of the dataset repository.

    Kept separate from filter_jobs.DatasetRepository (read-side, `get`)
    on purpose: each use case declares only what it needs. The concrete
    SqlAlchemyDatasetRepository satisfies both.
    """

    def add(self, dataset: SeismicDataset) -> SeismicDataset: ...


class ImportDatasetUseCase:
    """Import a SEG-Y as a persisted SeismicDataset.

    Coordinates two ports and nothing else: read the file's metadata,
    then persist the resulting entity, returning the entity *as the
    repository returned it* -- i.e. carrying the id SQLite assigned. If
    reading fails, add() is never called (no half-imported row); if
    persisting fails, the error propagates unchanged to the caller.

    Framework-free: no PyQt5, no SQLAlchemy. Callable so it can be
    injected anywhere a `(source_path, name) -> SeismicDataset` function
    is expected -- including SegyImportWorker.
    """

    def __init__(
        self,
        read_metadata: DatasetMetadataReader,
        datasets: DatasetWriteRepository,
    ) -> None:
        self._read_metadata = read_metadata
        self._datasets = datasets

    def __call__(self, source_path: str, name: str) -> SeismicDataset:
        unpersisted = self._read_metadata(source_path, name)
        return self._datasets.add(unpersisted)
