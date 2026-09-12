from dataclasses import replace

import pytest

from giecar_seismic.application.dataset_import import ImportDatasetUseCase
from giecar_seismic.domain.dataset import SeismicDataset


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


class RaisingDatasetRepository:
    def add(self, dataset: SeismicDataset) -> SeismicDataset:
        raise RuntimeError("database is locked")


def test_use_case_reads_metadata_with_the_given_path_and_name():
    calls: list[tuple[str, str]] = []
    unpersisted = _unpersisted_dataset()

    def read_metadata(source_path: str, name: str) -> SeismicDataset:
        calls.append((source_path, name))
        return unpersisted

    use_case = ImportDatasetUseCase(read_metadata, RecordingDatasetRepository())

    use_case("/data/survey.segy", "survey")

    assert calls == [("/data/survey.segy", "survey")]


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
