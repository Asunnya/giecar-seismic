import inspect

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.ui.workers import SegyImportWorker


def _fake_dataset() -> SeismicDataset:
    return SeismicDataset(
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=401,
        n_crosslines=720,
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


def test_worker_defaults_to_the_real_import_segy_dataset():
    from giecar_seismic.infrastructure.segy.dataset_importer import (
        import_segy_dataset,
    )

    worker = SegyImportWorker("/data/survey.segy", "survey")

    assert worker._importer is import_segy_dataset
