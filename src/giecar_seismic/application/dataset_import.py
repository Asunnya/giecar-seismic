import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from giecar_seismic.domain.dataset import SeismicDataset, SourceFingerprint

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

    def find_by_source_path(self, source_path: str) -> SeismicDataset | None: ...


class ImportDatasetUseCase:
    """Import a SEG-Y as a persisted SeismicDataset.

    A dataset's identity is its resolved absolute source path *plus* the
    file's fingerprint (size, mtime): importing a path that is already
    persisted with the same fingerprint returns the existing entity
    without re-reading headers or inserting a duplicate row (so jobs and
    the geometry index of that file stay attached to one dataset). If
    the file changed since -- or the stored row has no fingerprint -- a
    new dataset is imported and the old one keeps its history; its
    metadata is never rewritten under jobs computed against the old
    file. Otherwise:
    read the file's metadata, then persist the resulting entity,
    returning the entity *as the repository returned it* -- i.e.
    carrying the id SQLite assigned. If reading fails, add() is never
    called (no half-imported row); if persisting fails, the error
    propagates unchanged to the caller.

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
        # Relative paths, `..` and symlinks all collapse to one identity.
        resolved = str(Path(source_path).resolve())
        fingerprint = read_source_fingerprint(resolved)
        existing = self._datasets.find_by_source_path(resolved)
        if (
            existing is not None
            and fingerprint is not None
            and existing.source_fingerprint == fingerprint
        ):
            return existing
        unpersisted = self._read_metadata(resolved, name)
        return self._datasets.add(replace(unpersisted, source_fingerprint=fingerprint))


def read_source_fingerprint(path: str | Path) -> SourceFingerprint | None:
    """Size + mtime of the file, or None if it cannot be stat'ed (the
    metadata reader then raises its own, more specific error)."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return SourceFingerprint(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
