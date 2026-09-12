import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from giecar_seismic.application.dataset_import import ImportDatasetUseCase
from giecar_seismic.application.filter_jobs import FilterJobService
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
)
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset
from giecar_seismic.infrastructure.segy.reader import open_dataset_reader
from giecar_seismic.infrastructure.storage.hdf5_writer import make_hdf5_writer_factory
from giecar_seismic.ui.main_window import MainWindow

# Plain defaults for now -- a settings mechanism is out of scope until
# there's more than one thing to configure.
APP_DIR = Path.home() / ".giecar-seismic"
DEFAULT_DATABASE_PATH = APP_DIR / "giecar.sqlite"
DEFAULT_OUTPUTS_DIR = APP_DIR / "outputs"


def main() -> int:
    # Composition root: the only place that knows about SQLite/SQLAlchemy,
    # segyio and h5py together. The UI receives plain callables/objects
    # (ImportDatasetUseCase, FilterJobService) and never touches any of
    # those libraries itself.
    APP_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_sqlite_engine(DEFAULT_DATABASE_PATH)
    # create_all is not a migration: it only adds missing tables. A
    # database created with an older schema must be recreated explicitly.
    create_schema(engine)
    session_factory = make_session_factory(engine)
    datasets = SqlAlchemyDatasetRepository(session_factory)
    jobs = SqlAlchemyJobRepository(session_factory)

    import_dataset = ImportDatasetUseCase(import_segy_dataset, datasets)
    service = FilterJobService(
        datasets=datasets,
        jobs=jobs,
        reader_factory=open_dataset_reader,
        writer_factory=make_hdf5_writer_factory(DEFAULT_OUTPUTS_DIR),
    )

    app = QApplication(sys.argv)
    window = MainWindow(service=service, dataset_importer=import_dataset)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
