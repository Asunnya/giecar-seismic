import inspect
from dataclasses import replace

import pytest

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.ui.workers import SegyImportWorker


def _fake_dataset() -> SeismicDataset:
    return SeismicDataset(
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=401,
        n_crosslines=720,
        n_traces=288694,
        n_samples=850,
        sample_rate_ms=4.0,
    )


def test_worker_never_imports_or_touches_qtwidgets():
    # SegyImportWorker must be usable with nothing but plain Python
    # objects: no QApplication, no QWidget, no real SEG-Y file.
    import giecar_seismic.ui.workers as workers_module

    source = inspect.getsource(workers_module)
    assert "PyQt5.QtWidgets" not in source


def test_worker_emits_succeeded_with_the_dataset_from_the_injected_importer():
    dataset = _fake_dataset()
    calls: list[tuple[str, str]] = []

    def fake_importer(source_path: str, name: str) -> SeismicDataset:
        calls.append((source_path, name))
        return dataset

    worker = SegyImportWorker("/data/survey.segy", "survey", importer=fake_importer)

    succeeded: list[SeismicDataset] = []
    failures: list[str] = []
    worker.succeeded.connect(succeeded.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert calls == [("/data/survey.segy", "survey")]
    assert succeeded == [dataset]
    assert failures == []


def test_worker_emits_failed_when_the_importer_raises():
    def raising_importer(source_path: str, name: str) -> SeismicDataset:
        raise OSError("not a valid SEG-Y file")

    worker = SegyImportWorker("/data/broken.segy", "broken", importer=raising_importer)

    succeeded: list[SeismicDataset] = []
    failures: list[str] = []
    worker.succeeded.connect(succeeded.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert succeeded == []
    assert failures == ["not a valid SEG-Y file"]


def test_worker_requires_an_importer_and_knows_no_infrastructure():
    # The importer is injected -- the worker has no default wired to
    # segyio, SQLAlchemy or any repository, so it can't reach either on
    # its own.
    import giecar_seismic.ui.workers as workers_module

    source = inspect.getsource(workers_module)
    assert "sqlalchemy" not in source.lower()
    assert "import_segy_dataset" not in source
    with pytest.raises(TypeError):
        SegyImportWorker("/data/survey.segy", "survey")  # type: ignore[call-arg]


def test_worker_delivers_the_persisted_dataset_with_its_id():
    persisted = replace(_fake_dataset(), id=7)
    worker = SegyImportWorker(
        "/data/survey.segy", "survey", importer=lambda path, name: persisted
    )
    succeeded: list[SeismicDataset] = []
    worker.succeeded.connect(succeeded.append)

    worker.run()

    assert succeeded == [persisted]
    assert succeeded[0].id == 7
