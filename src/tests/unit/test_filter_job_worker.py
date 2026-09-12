import inspect

import numpy as np

from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus
from giecar_seismic.ui.workers import FilterJobWorker


class FakeDatasetRepository:
    def __init__(self, datasets: list[SeismicDataset]):
        self._datasets = {dataset.id: dataset for dataset in datasets}

    def get(self, dataset_id: int) -> SeismicDataset | None:
        return self._datasets.get(dataset_id)


class FakeJobRepository:
    def __init__(self) -> None:
        self._jobs: dict[int, Job] = {}
        self._next_id = 1

    def add(self, job: Job) -> Job:
        job.id = self._next_id
        self._jobs[job.id] = job
        self._next_id += 1
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job: Job) -> None:
        assert job.id is not None
        self._jobs[job.id] = job

    def list(self, dataset_id=None, status=None):
        return list(self._jobs.values())


class FakeTraceReader:
    def __init__(self, traces: np.ndarray):
        self.trace_count = len(traces)
        self._traces = traces

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        return self._traces[start:stop]

    def close(self) -> None:
        pass


class FakeTraceWriter:
    output_path = "/fake/output.h5"

    def __init__(self) -> None:
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))

    def finalize(self) -> None:
        self.finalized = True

    def close(self) -> None:
        pass


class RaisingWriteChunkTraceWriter:
    output_path = "/fake/output.h5"

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        raise OSError("disk full")

    def finalize(self) -> None:
        pass

    def close(self) -> None:
        pass


def _build_service(
    dataset: SeismicDataset, reader, writer, chunk_size: int = 2
) -> FilterJobService:
    return FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda ds: reader,
        writer_factory=lambda job, dataset: writer,
        chunk_size=chunk_size,
    )


N_SAMPLES = 64  # large enough for sosfiltfilt's padlen at order=4


def _dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=4,
        n_samples=N_SAMPLES,
        sample_rate_ms=4.0,
    )


def test_worker_emits_progress_without_touching_any_widget():
    # FilterJobWorker must be usable with nothing but plain Python
    # objects: no QApplication, no QWidget, no real SEG-Y/HDF5. This also
    # documents, structurally, that the module never imports QtWidgets.
    import giecar_seismic.ui.workers as workers_module

    source = inspect.getsource(workers_module)
    assert "PyQt5.QtWidgets" not in source

    dataset = _dataset()
    traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=2)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    worker = FilterJobWorker(service, job.id, CooperativeCancelToken())

    progresses: list[float] = []
    completed_jobs: list[Job] = []
    worker.progress.connect(progresses.append)
    worker.completed.connect(completed_jobs.append)

    worker.run()

    assert progresses == [50, 100]
    assert len(completed_jobs) == 1
    assert completed_jobs[0].status is JobStatus.COMPLETED
    assert writer.finalized is True


def test_worker_emits_failed_when_the_job_ends_in_failed_status():
    dataset = _dataset()
    reader = FakeTraceReader(np.zeros((4, N_SAMPLES), dtype=np.float32))
    writer = RaisingWriteChunkTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=2)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    worker = FilterJobWorker(service, job.id, CooperativeCancelToken())

    failures: list[str] = []
    completed_jobs: list[Job] = []
    worker.failed.connect(failures.append)
    worker.completed.connect(completed_jobs.append)

    worker.run()

    assert failures == ["disk full"]
    assert completed_jobs == []


def test_worker_emits_cancelled_when_the_token_is_already_cancelled():
    dataset = _dataset()
    reader = FakeTraceReader(np.zeros((4, N_SAMPLES), dtype=np.float32))
    writer = FakeTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=2)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    cancel_token = CooperativeCancelToken()
    cancel_token.request_cancel()
    worker = FilterJobWorker(service, job.id, cancel_token)

    cancelled_jobs: list[Job] = []
    completed_jobs: list[Job] = []
    worker.cancelled.connect(cancelled_jobs.append)
    worker.completed.connect(completed_jobs.append)

    worker.run()

    assert len(cancelled_jobs) == 1
    assert cancelled_jobs[0].status is JobStatus.CANCELLED
    assert completed_jobs == []
    assert writer.finalized is False


def test_worker_emits_failed_when_run_filter_job_raises_before_returning_a_job():
    dataset = _dataset()
    # A service with no reader_factory/writer_factory configured makes
    # run_filter_job() raise RuntimeError directly -- a path distinct
    # from a job ending in FAILED status -- and this must still reach the
    # GUI via the `failed` signal rather than propagating out of the
    # worker thread and crashing it.
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    worker = FilterJobWorker(service, job.id, CooperativeCancelToken())

    failures: list[str] = []
    worker.failed.connect(failures.append)

    worker.run()

    assert len(failures) == 1
    assert "reader_factory" in failures[0] or "writer_factory" in failures[0]
