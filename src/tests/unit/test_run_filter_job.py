import numpy as np
import pytest

from giecar_seismic.application.filter_jobs import FilterJobService
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus


class FakeDatasetRepository:
    def __init__(self, datasets: list[SeismicDataset] | None = None):
        self._datasets = {dataset.id: dataset for dataset in datasets or []}

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
        self.traces = traces
        self.trace_count = len(traces)
        self.read_calls: list[tuple[int, int]] = []

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        self.read_calls.append((start, stop))
        return self.traces[start:stop]


class FakeTraceWriter:
    def __init__(self) -> None:
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))

    def finalize(self) -> None:
        self.finalized = True


class FakeCancelToken:
    def __init__(self, cancel_after: int | None = None):
        self.calls = 0
        self._cancel_after = cancel_after

    def is_cancelled(self) -> bool:
        self.calls += 1
        if self._cancel_after is None:
            return False
        return self.calls > self._cancel_after


@pytest.fixture
def dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=401,
        n_crosslines=720,
        n_samples=850,
        sample_rate_ms=4.0,  # nyquist_hz == 125.0
    )


def _build_service(
    dataset: SeismicDataset,
    reader: FakeTraceReader,
    writer: FakeTraceWriter,
    chunk_size: int = 4,
) -> FilterJobService:
    return FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda ds: reader,
        writer_factory=lambda job: writer,
        chunk_size=chunk_size,
    )


def test_run_filter_job_completes_and_writes_all_chunks(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=4)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    progresses: list[float] = []
    service.run_filter_job(
        job.id, progress_callback=progresses.append, cancel_token=FakeCancelToken()
    )

    completed = service.get_job_status(job.id)
    assert completed.status is JobStatus.COMPLETED
    assert writer.finalized is True
    assert reader.read_calls == [(0, 4), (4, 8), (8, 10)]
    assert sum(chunk.shape[0] for _, chunk in writer.written) == 10
    assert progresses == sorted(progresses)
    assert progresses[-1] == 100


def test_run_filter_job_stops_cooperatively_when_cancelled_mid_run(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=4)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    # is_cancelled() is checked once before each chunk: call 1 (before chunk
    # [0:4]) is not cancelled yet, call 2 (before chunk [4:8]) is.
    cancel_token = FakeCancelToken(cancel_after=1)
    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=cancel_token
    )

    cancelled = service.get_job_status(job.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert reader.read_calls == [(0, 4)]
    assert len(writer.written) == 1
    assert writer.finalized is False


class RaisingTraceReader:
    trace_count = 10

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        raise OSError("segyio read error")


def test_run_filter_job_fails_the_job_when_reading_raises(dataset):
    writer = FakeTraceWriter()
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda ds: RaisingTraceReader(),
        writer_factory=lambda job: writer,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error_message == "segyio read error"
    assert writer.finalized is False
