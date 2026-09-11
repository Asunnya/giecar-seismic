from collections.abc import Callable
from typing import Protocol

import numpy as np

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter
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


class TraceReader(Protocol):
    trace_count: int

    def read_chunk(self, start: int, stop: int) -> np.ndarray: ...


class TraceWriter(Protocol):
    def write_chunk(self, start: int, chunk: np.ndarray) -> None: ...
    def finalize(self) -> None: ...


class CancelToken(Protocol):
    def is_cancelled(self) -> bool: ...


ReaderFactory = Callable[[SeismicDataset], TraceReader]
WriterFactory = Callable[[Job], TraceWriter]
ProgressCallback = Callable[[float], None]


class DatasetNotFoundError(Exception):
    pass


class JobNotFoundError(Exception):
    pass


class InvalidFilterParametersError(ValueError):
    pass


MIN_FILTER_ORDER = 2
MAX_FILTER_ORDER = 8

# 256 traces/chunk: notebooks/01_inspect_segy.ipynb's exploratory benchmark
# found this in the flat/best region for the sample survey. Not a proof of
# optimality (page-cache effects), just a reasonable default.
DEFAULT_CHUNK_SIZE = 256


class FilterJobService:
    def __init__(
        self,
        datasets: DatasetRepository,
        jobs: JobRepository,
        reader_factory: ReaderFactory | None = None,
        writer_factory: WriterFactory | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self._datasets = datasets
        self._jobs = jobs
        self._reader_factory = reader_factory
        self._writer_factory = writer_factory
        self._chunk_size = chunk_size

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

    def run_filter_job(
        self,
        job_id: int,
        progress_callback: ProgressCallback,
        cancel_token: CancelToken,
    ) -> Job:
        if self._reader_factory is None or self._writer_factory is None:
            raise RuntimeError(
                "FilterJobService requires reader_factory and writer_factory "
                "to run a filter job"
            )

        job = self.get_job_status(job_id)
        dataset = self._datasets.get(job.dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"dataset {job.dataset_id} not found")

        job.start()
        self._jobs.update(job)

        reader = self._reader_factory(dataset)
        writer = self._writer_factory(job)

        try:
            trace_count = reader.trace_count
            for start in range(0, trace_count, self._chunk_size):
                if cancel_token.is_cancelled():
                    job.cancel()
                    return job

                stop = min(start + self._chunk_size, trace_count)
                chunk = reader.read_chunk(start, stop)
                filtered = apply_lowpass_filter(
                    chunk, job.cutoff_hz, job.order, dataset.sample_rate_ms
                )
                writer.write_chunk(start, filtered)
                progress_callback(round(100 * stop / trace_count))

            writer.finalize()
            job.complete()
        except Exception as exc:  # noqa: BLE001 -- any failure must FAIL the job, not crash the worker
            job.fail(str(exc))
        finally:
            self._jobs.update(job)

        return job
