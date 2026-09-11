from typing import Protocol

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus


class DatasetRepository(Protocol):
    def get(self, dataset_id: int) -> SeismicDataset | None: ...


class JobRepository(Protocol):
    def add(self, job: Job) -> Job: ...
    def get(self, job_id: int) -> Job | None: ...
    def update(self, job: Job) -> None: ...
    def list(
        self, dataset_id: int | None = None, status: JobStatus | None = None
    ) -> list[Job]: ...


class DatasetNotFoundError(Exception):
    pass


class JobNotFoundError(Exception):
    pass


class InvalidFilterParametersError(ValueError):
    pass


MIN_FILTER_ORDER = 2
MAX_FILTER_ORDER = 8


class FilterJobService:
    def __init__(self, datasets: DatasetRepository, jobs: JobRepository):
        self._datasets = datasets
        self._jobs = jobs

    def create_filter_job(self, dataset_id: int, cutoff_hz: float, order: int) -> Job:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"dataset {dataset_id} not found")

        if not (0 < cutoff_hz < dataset.nyquist_hz):
            raise InvalidFilterParametersError(
                f"cutoff_hz must be between 0 and the dataset's Nyquist "
                f"frequency ({dataset.nyquist_hz} Hz), got {cutoff_hz}"
            )

        if not (MIN_FILTER_ORDER <= order <= MAX_FILTER_ORDER):
            raise InvalidFilterParametersError(
                f"order must be between {MIN_FILTER_ORDER} and "
                f"{MAX_FILTER_ORDER}, got {order}"
            )

        job = Job(dataset_id=dataset_id, cutoff_hz=cutoff_hz, order=order)
        return self._jobs.add(job)

    def list_jobs(
        self, dataset_id: int | None = None, status: JobStatus | None = None
    ) -> list[Job]:
        return self._jobs.list(dataset_id=dataset_id, status=status)

    def get_job_status(self, job_id: int) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"job {job_id} not found")
        return job

    def cancel_job(self, job_id: int) -> Job:
        job = self.get_job_status(job_id)
        job.cancel()
        self._jobs.update(job)
        return job
