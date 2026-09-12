import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget

from giecar_seismic.application.dataset_import import ImportDatasetUseCase
from giecar_seismic.application.filter_jobs import FilterJobService
from giecar_seismic.application.geometry_index import BuildGeometryIndexUseCase
from giecar_seismic.application.seismic_viewer import SeismicViewerService, ViewerTarget
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyGeometryRepository,
    SqlAlchemyJobRepository,
)
from giecar_seismic.infrastructure.logging.job_execution_logger import (
    FileJobExecutionLogger,
    read_job_log,
)
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset
from giecar_seismic.infrastructure.segy.reader import (
    iter_trace_header_batches,
    open_dataset_reader,
)
from giecar_seismic.infrastructure.storage.hdf5_reader import open_job_output_reader
from giecar_seismic.infrastructure.storage.hdf5_writer import (
    make_hdf5_writer_factory,
    open_hdf5_resume_writer,
)
from giecar_seismic.ui.main_window import MainWindow
from giecar_seismic.ui.seismic_viewer import SeismicViewer

# Plain defaults for now -- a settings mechanism is out of scope until
# there's more than one thing to configure.
APP_DIR = Path.home() / ".giecar-seismic"
DEFAULT_DATABASE_PATH = APP_DIR / "giecar.sqlite"
DEFAULT_OUTPUTS_DIR = APP_DIR / "outputs"
DEFAULT_LOGS_DIR = APP_DIR / "logs"  # one append-only job_<id>.log per job


def parse_filter_process_count(
    raw_value: str | None, *, available_cpus: int | None = None
) -> int:
    """Parse the opt-in process count before any GUI resource is created."""
    value = (raw_value or "").strip()
    if not value:
        return 1
    try:
        processes = int(value)
    except ValueError as exc:
        raise ValueError(
            "GIECAR_FILTER_PROCESSES must be an integer greater than or equal to 1"
        ) from exc
    if processes < 1:
        raise ValueError(
            "GIECAR_FILTER_PROCESSES must be an integer greater than or equal to 1"
        )
    cpu_limit = max(1, available_cpus or os.cpu_count() or 1)
    if processes > cpu_limit:
        raise ValueError(
            "GIECAR_FILTER_PROCESSES cannot exceed the available CPUs "
            f"({cpu_limit}), got {processes}"
        )
    return processes


def main() -> int:
    # Composition root: the only place that knows about SQLite/SQLAlchemy,
    # segyio and h5py together. The UI receives plain callables/objects
    # (ImportDatasetUseCase, FilterJobService) and never touches any of
    # those libraries itself.
    parallel_workers = parse_filter_process_count(
        os.environ.get("GIECAR_FILTER_PROCESSES")
    )
    APP_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
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
        resume_writer_factory=open_hdf5_resume_writer,
        parallel_workers=parallel_workers,
        execution_logger=FileJobExecutionLogger(DEFAULT_LOGS_DIR),
    )
    # A previous process may have died mid-run (the scenario Resume exists
    # for): jobs it left RUNNING become CANCELLED so their HDF5 checkpoint
    # is reachable through Resume. Before any window exists.
    service.recover_interrupted_jobs()

    # Viewer: geometry index (built lazily, in bounded batches, on first
    # open) + selective SEG-Y/HDF5 readers -- the same physical index
    # addresses both, since the pipeline preserves trace order.
    geometry = SqlAlchemyGeometryRepository(session_factory)
    build_geometry_index = BuildGeometryIndexUseCase(
        iter_trace_header_batches, geometry
    )
    viewer_service = SeismicViewerService(
        datasets=datasets,
        jobs=jobs,
        geometry=geometry,
        original_reader_factory=open_dataset_reader,
        filtered_reader_factory=open_job_output_reader,
    )

    def open_viewer(target: ViewerTarget, parent: QWidget) -> QWidget:
        return SeismicViewer(viewer_service, build_geometry_index, target, parent)

    app = QApplication(sys.argv)
    window = MainWindow(
        service=service,
        dataset_importer=import_dataset,
        open_viewer=open_viewer,
        read_job_log=lambda job_id: read_job_log(DEFAULT_LOGS_DIR, job_id),
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
