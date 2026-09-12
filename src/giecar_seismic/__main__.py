import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from giecar_seismic.application.dataset_import import ImportDatasetUseCase
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
)
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset
from giecar_seismic.ui.main_window import MainWindow

# Plain default for now -- a settings mechanism is out of scope until
# there's more than one thing to configure.
DEFAULT_DATABASE_PATH = Path.home() / ".giecar-seismic" / "giecar.sqlite"


def main() -> int:
    # Composition root: the only place that knows about SQLite, SQLAlchemy
    # and segyio together. The UI receives a plain callable.
    #
    # Only the dataset import is composed. `service` (FilterJobService)
    # is still not: the SEG-Y/HDF5 reader/writer factories it needs for a
    # real end-to-end run aren't wired yet, and faking them here would
    # hide that gap behind a demo. The window opens with service=None,
    # visibly incomplete (Run Filter stays disabled) until that lands.
    DEFAULT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_sqlite_engine(DEFAULT_DATABASE_PATH)
    create_schema(engine)  # no migrations in this challenge: idempotent create_all
    datasets = SqlAlchemyDatasetRepository(make_session_factory(engine))
    import_dataset = ImportDatasetUseCase(import_segy_dataset, datasets)

    app = QApplication(sys.argv)
    window = MainWindow(dataset_importer=import_dataset)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
