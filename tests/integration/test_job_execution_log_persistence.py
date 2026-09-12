"""Real SQLite + SEG-Y + HDF5 + a real log directory: the execution log of a
job survives closing and recomposing the application, and multiprocessing
subprocesses never write to it."""

import h5py
import numpy as np

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter
from giecar_seismic.application.dataset_import import ImportDatasetUseCase
from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.domain.job import JobStatus
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
)
from giecar_seismic.infrastructure.logging.job_execution_logger import (
    FileJobExecutionLogger,
    read_job_log,
)
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset
from giecar_seismic.infrastructure.segy.reader import open_dataset_reader
from giecar_seismic.infrastructure.storage.hdf5_writer import (
    make_hdf5_writer_factory,
    open_hdf5_resume_writer,
)
from tests.e2e.test_end_to_end_filter_pipeline import FOOTPRINTS, write_segy


def compose(root, *, chunk_size=2, parallel_workers=1):
    """Mirror of __main__ with temporary paths; call again to 'restart'."""
    engine = create_sqlite_engine(root / "giecar.sqlite")
    create_schema(engine)
    sessions = make_session_factory(engine)
    datasets = SqlAlchemyDatasetRepository(sessions)
    jobs = SqlAlchemyJobRepository(sessions)
    service = FilterJobService(
        datasets=datasets,
        jobs=jobs,
        reader_factory=open_dataset_reader,
        writer_factory=make_hdf5_writer_factory(root / "outputs"),
        resume_writer_factory=open_hdf5_resume_writer,
        chunk_size=chunk_size,
        parallel_workers=parallel_workers,
        execution_logger=FileJobExecutionLogger(root / "logs"),
    )
    return engine, ImportDatasetUseCase(import_segy_dataset, datasets), service


def _events(text: str) -> list[str]:
    return [line.split(" ")[3] for line in text.strip().splitlines()]


def test_log_survives_restart_and_resume_appends_to_the_same_file(tmp_path):
    write_segy(tmp_path / "survey.segy", FOOTPRINTS[0], 64, 4000)
    engine, importer, service = compose(tmp_path)
    dataset = importer(str(tmp_path / "survey.segy"), "survey")
    job = service.create_filter_job(dataset.id, 30, 4)
    token = CooperativeCancelToken()
    job = service.run_filter_job(job.id, lambda _: token.request_cancel(), token)
    assert job.status is JobStatus.CANCELLED
    engine.dispose()

    before = read_job_log(tmp_path / "logs", job.id)
    assert _events(before) == ["JOB_CREATED", "RUN_STARTED", "JOB_CANCELLED"]

    # "restart": a fresh composition over the same directory
    engine, _, service = compose(tmp_path)
    done = service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    engine.dispose()
    assert done.status is JobStatus.COMPLETED

    after = read_job_log(tmp_path / "logs", job.id)
    assert after.startswith(before)  # append-only, nothing truncated
    assert _events(after) == [
        "JOB_CREATED",
        "RUN_STARTED",
        "JOB_CANCELLED",
        "RESUME_STARTED",
        "JOB_COMPLETED",
    ]
    assert sorted(p.name for p in (tmp_path / "logs").iterdir()) == [
        f"job_{job.id}.log"
    ]


def test_multiprocessing_run_logs_once_from_the_coordinator_only(tmp_path):
    amplitudes = write_segy(tmp_path / "survey.segy", FOOTPRINTS[0], 64, 4000)
    engine, importer, service = compose(tmp_path, parallel_workers=2)
    dataset = importer(str(tmp_path / "survey.segy"), "survey")
    job = service.create_filter_job(dataset.id, 30, 4)
    done = service.run_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    engine.dispose()

    assert done.status is JobStatus.COMPLETED
    assert sorted(p.name for p in (tmp_path / "logs").iterdir()) == [
        f"job_{job.id}.log"
    ]
    assert _events(read_job_log(tmp_path / "logs", job.id)) == [
        "JOB_CREATED",
        "RUN_STARTED",
        "JOB_COMPLETED",
    ]
    with h5py.File(done.output_path, "r") as file:
        np.testing.assert_allclose(
            file["traces"][:],
            apply_lowpass_filter(amplitudes, 30, 4, 4).astype(np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
