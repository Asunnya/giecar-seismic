from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus
from giecar_seismic.infrastructure.database.models import DatasetModel, JobModel

# Every repository method follows the same session lifecycle:
#
#   open session -> (begin transaction) -> operation -> commit on success /
#   rollback on exception -> convert ORM rows to domain objects -> close.
#
# `with session_factory() as session, session.begin():` does exactly that:
# begin() commits when the block exits normally and rolls back if it
# raises; the outer `with` closes the session either way. Infrastructure
# errors (IntegrityError, OperationalError, ...) propagate to the caller --
# nothing here swallows them. No Session is ever kept as module or
# instance state; each call gets its own.


def _dataset_to_domain(model: DatasetModel) -> SeismicDataset:
    return SeismicDataset(
        id=model.id,
        name=model.name,
        source_path=model.source_path,
        n_inlines=model.n_inlines,
        n_crosslines=model.n_crosslines,
        n_traces=model.n_traces,
        n_samples=model.n_samples,
        sample_rate_ms=model.sample_rate_ms,
        created_at=model.created_at,
    )


def _dataset_to_model(dataset: SeismicDataset) -> DatasetModel:
    # id is intentionally left unset for a new dataset (dataset.id is
    # None): SQLite's autoincrement primary key assigns it on flush.
    return DatasetModel(
        id=dataset.id,
        name=dataset.name,
        source_path=dataset.source_path,
        n_inlines=dataset.n_inlines,
        n_crosslines=dataset.n_crosslines,
        n_traces=dataset.n_traces,
        n_samples=dataset.n_samples,
        sample_rate_ms=dataset.sample_rate_ms,
        created_at=dataset.created_at,
    )


def _job_to_domain(model: JobModel) -> Job:
    return Job(
        id=model.id,
        dataset_id=model.dataset_id,
        cutoff_hz=model.cutoff_hz,
        order=model.order,
        status=JobStatus[model.status],
        error_message=model.error_message,
        progress=model.progress,
        output_path=model.output_path,
        created_at=model.created_at,
        started_at=model.started_at,
        finished_at=model.finished_at,
    )


def _job_to_model(job: Job) -> JobModel:
    return JobModel(
        id=job.id,
        dataset_id=job.dataset_id,
        cutoff_hz=job.cutoff_hz,
        order=job.order,
        status=job.status.name,
        error_message=job.error_message,
        progress=job.progress,
        output_path=job.output_path,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


class SqlAlchemyDatasetRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add(self, dataset: SeismicDataset) -> SeismicDataset:
        model = _dataset_to_model(dataset)
        with self._session_factory() as session, session.begin():
            session.add(model)
            session.flush()  # assigns the autoincrement id before conversion
            return _dataset_to_domain(model)

    def get(self, dataset_id: int) -> SeismicDataset | None:
        with self._session_factory() as session:
            model = session.get(DatasetModel, dataset_id)
            return _dataset_to_domain(model) if model is not None else None


class SqlAlchemyJobRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add(self, job: Job) -> Job:
        model = _job_to_model(job)
        with self._session_factory() as session, session.begin():
            session.add(model)
            session.flush()
            return _job_to_domain(model)

    def get(self, job_id: int) -> Job | None:
        with self._session_factory() as session:
            model = session.get(JobModel, job_id)
            return _job_to_domain(model) if model is not None else None

    def update(self, job: Job) -> None:
        if job.id is None:
            raise ValueError("cannot update a job that has no id (never persisted)")
        with self._session_factory() as session, session.begin():
            model = session.get(JobModel, job.id)
            if model is None:
                raise LookupError(f"job {job.id} not found")
            model.dataset_id = job.dataset_id
            model.cutoff_hz = job.cutoff_hz
            model.order = job.order
            model.status = job.status.name
            model.error_message = job.error_message
            model.progress = job.progress
            model.output_path = job.output_path
            model.started_at = job.started_at
            model.finished_at = job.finished_at

    def list(
        self, dataset_id: int | None = None, status: JobStatus | None = None
    ) -> list[Job]:
        statement = select(JobModel)
        if dataset_id is not None:
            statement = statement.where(JobModel.dataset_id == dataset_id)
        if status is not None:
            statement = statement.where(JobModel.status == status.name)
        statement = statement.order_by(JobModel.id)
        with self._session_factory() as session:
            return [_job_to_domain(model) for model in session.scalars(statement)]
