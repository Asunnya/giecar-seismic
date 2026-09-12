import os
from dataclasses import replace
from pathlib import Path

import pytest

from giecar_seismic.application.dataset_import import (
    ImportDatasetUseCase,
    read_source_fingerprint,
)
from giecar_seismic.domain.dataset import SeismicDataset, SourceFingerprint


def _unpersisted_dataset() -> SeismicDataset:
    return SeismicDataset(
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=3,
        n_crosslines=3,
        n_traces=8,
        n_samples=5,
        sample_rate_ms=4.0,
    )


class RecordingDatasetRepository:
    """Assigns an id on add() and records what it was asked to persist --
    the use case must hand back the entity *the repository returned*, not
    the unpersisted one it received from the metadata reader.
    """

    def __init__(self) -> None:
        self.added: list[SeismicDataset] = []

    def add(self, dataset: SeismicDataset) -> SeismicDataset:
        self.added.append(dataset)
        return replace(dataset, id=42)

    def find_by_source_path(self, source_path: str) -> SeismicDataset | None:
        return None


class RaisingDatasetRepository:
    def add(self, dataset: SeismicDataset) -> SeismicDataset:
        raise RuntimeError("database is locked")

    def find_by_source_path(self, source_path: str) -> SeismicDataset | None:
        return None


def test_use_case_reads_metadata_with_the_given_path_and_name():
    calls: list[tuple[str, str]] = []
    unpersisted = _unpersisted_dataset()

    def read_metadata(source_path: str, name: str) -> SeismicDataset:
        calls.append((source_path, name))
        return unpersisted

    use_case = ImportDatasetUseCase(read_metadata, RecordingDatasetRepository())

    use_case("/data/survey.segy", "survey")

    # The reader receives the path the user chose, in its resolved form --
    # which is platform-specific (a drive letter is prepended on Windows).
    assert calls == [(str(Path("/data/survey.segy").resolve()), "survey")]


def test_use_case_persists_the_imported_entity_and_returns_the_persisted_one():
    unpersisted = _unpersisted_dataset()
    repository = RecordingDatasetRepository()
    use_case = ImportDatasetUseCase(lambda path, name: unpersisted, repository)

    persisted = use_case("/data/survey.segy", "survey")

    assert repository.added == [unpersisted]
    assert unpersisted.id is None
    assert persisted.id == 42
    assert persisted.n_traces == 8
    assert persisted.name == "survey"


def test_metadata_reader_failure_never_reaches_the_repository():
    repository = RecordingDatasetRepository()

    def failing_reader(source_path: str, name: str) -> SeismicDataset:
        raise OSError("not a valid SEG-Y file")

    use_case = ImportDatasetUseCase(failing_reader, repository)

    with pytest.raises(OSError, match="not a valid SEG-Y file"):
        use_case("/data/broken.segy", "broken")

    assert repository.added == []


def test_repository_failure_propagates_to_the_caller():
    use_case = ImportDatasetUseCase(
        lambda path, name: _unpersisted_dataset(), RaisingDatasetRepository()
    )

    with pytest.raises(RuntimeError, match="database is locked"):
        use_case("/data/survey.segy", "survey")


# --- reuse by source path ---------------------------------------------------


class LookupDatasetRepository(RecordingDatasetRepository):
    """Repository that already holds datasets, keyed by source path."""

    def __init__(self, existing: list[SeismicDataset]) -> None:
        super().__init__()
        self.existing = list(existing)
        self.lookups: list[str] = []

    def find_by_source_path(self, source_path: str) -> SeismicDataset | None:
        self.lookups.append(source_path)
        return next((d for d in self.existing if d.source_path == source_path), None)


def _touch(path, content: bytes = b"segy") -> str:
    path.write_bytes(content)
    return str(path)


def test_importing_an_already_imported_unchanged_file_returns_the_existing_dataset_without_reading(
    tmp_path,
):
    path = _touch(tmp_path / "survey.segy")
    existing = replace(
        _unpersisted_dataset(),
        id=7,
        source_path=path,
        source_fingerprint=read_source_fingerprint(path),
    )
    repository = LookupDatasetRepository([existing])
    reads: list[str] = []

    def read_metadata(source_path: str, name: str) -> SeismicDataset:
        reads.append(source_path)
        return _unpersisted_dataset()

    use_case = ImportDatasetUseCase(read_metadata, repository)

    result = use_case(path, "survey")

    assert result == existing  # same id, same metadata -- nothing re-derived
    assert reads == []  # headers are never re-read
    assert repository.added == []  # no duplicate row


def test_importing_a_new_path_reads_and_persists_as_before():
    repository = LookupDatasetRepository([replace(_unpersisted_dataset(), id=7)])
    use_case = ImportDatasetUseCase(
        lambda path, name: replace(_unpersisted_dataset(), source_path=path, name=name),
        repository,
    )

    result = use_case("/data/other.segy", "other")

    assert result.id == 42
    assert [d.source_path for d in repository.added] == [
        str(Path("/data/other.segy").resolve())
    ]


def test_source_path_is_normalized_before_lookup_and_import(tmp_path):
    # The same file reached through a relative path, `..` or a symlink is
    # the same dataset: identity is the resolved absolute path.
    real = tmp_path / "surveys" / "survey.segy"
    real.parent.mkdir()
    real.write_bytes(b"segy")
    existing = replace(
        _unpersisted_dataset(),
        id=7,
        source_path=str(real),
        source_fingerprint=read_source_fingerprint(real),
    )
    repository = LookupDatasetRepository([existing])
    reads: list[str] = []

    def read_metadata(source_path: str, name: str) -> SeismicDataset:
        reads.append(source_path)
        return replace(_unpersisted_dataset(), source_path=source_path)

    use_case = ImportDatasetUseCase(read_metadata, repository)

    assert (
        use_case(str(tmp_path / "surveys" / ".." / "surveys" / "survey.segy"), "survey")
        == existing
    )
    assert reads == []
    assert repository.lookups == [str(real)]

    # Symlinks need a privilege on Windows: cover them where the OS allows.
    link = tmp_path / "link.segy"
    try:
        link.symlink_to(real)
    except OSError:
        link = None
    if link is not None:
        assert use_case(str(link), "survey") == existing
        assert reads == []
        assert repository.lookups == [str(real), str(real)]

    # a genuinely new file is imported under its resolved path, whatever was typed
    other = tmp_path / "surveys" / "other.segy"
    other.write_bytes(b"")
    use_case(str(tmp_path / "surveys" / ".." / "surveys" / "other.segy"), "other")
    assert reads == [str(other)]
    assert repository.added[-1].source_path == str(other)


# --- change detection: same path, different file --------------------------


def test_read_source_fingerprint_is_size_and_mtime_and_none_for_a_missing_file(
    tmp_path,
):
    path = tmp_path / "survey.segy"
    path.write_bytes(b"abc")
    os.utime(path, ns=(1_000, 1_700_000_000_000_000_000))

    assert read_source_fingerprint(path) == SourceFingerprint(
        size_bytes=3, mtime_ns=1_700_000_000_000_000_000
    )
    assert read_source_fingerprint(tmp_path / "missing.segy") is None


def test_persisted_dataset_carries_the_fingerprint_of_the_file_it_was_read_from(
    tmp_path,
):
    path = _touch(tmp_path / "survey.segy", b"12345")
    repository = LookupDatasetRepository([])
    use_case = ImportDatasetUseCase(lambda p, n: _unpersisted_dataset(), repository)

    use_case(path, "survey")

    assert repository.added[0].source_fingerprint == read_source_fingerprint(path)


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_a_changed_file_at_the_same_path_is_imported_as_a_new_dataset(tmp_path, change):
    path = tmp_path / "survey.segy"
    _touch(path, b"v1")
    old = replace(
        _unpersisted_dataset(),
        id=7,
        source_path=str(path),
        source_fingerprint=read_source_fingerprint(path),
    )
    repository = LookupDatasetRepository([old])
    if change == "size":
        _touch(path, b"v2 -- longer")
    else:
        # Above any filesystem's timestamp granularity (NTFS 100 ns, FAT 2 s):
        # a +1 ns bump would round back to the same mtime on Windows.
        os.utime(path, ns=(0, old.source_fingerprint.mtime_ns + 2_000_000_000))
    assert read_source_fingerprint(path) != old.source_fingerprint
    reads: list[str] = []

    def read_metadata(source_path: str, name: str) -> SeismicDataset:
        reads.append(source_path)
        return _unpersisted_dataset()

    use_case = ImportDatasetUseCase(read_metadata, repository)

    result = use_case(str(path), "survey")

    assert reads == [str(path)]  # re-read, since the old metadata may be stale
    assert result.id == 42 and result.id != old.id  # a new row, old history kept
    assert repository.added[0].source_fingerprint == read_source_fingerprint(path)


def test_an_existing_dataset_with_unknown_fingerprint_is_never_reused(tmp_path):
    # Rows imported before fingerprints existed: we cannot prove the file
    # is unchanged, so re-import rather than trust stale metadata.
    path = _touch(tmp_path / "survey.segy")
    old = replace(
        _unpersisted_dataset(), id=7, source_path=path, source_fingerprint=None
    )
    repository = LookupDatasetRepository([old])
    use_case = ImportDatasetUseCase(lambda p, n: _unpersisted_dataset(), repository)

    assert use_case(path, "survey").id == 42
