import threading

import numpy as np
import pytest

from giecar_seismic.application.filter_jobs import (
    CancellationWindowClosedError,
    CooperativeCancelToken,
    FilterJobService,
    JobAlreadyRunningError,
    TraceReader,
    TraceWriter,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus


class FakeDatasetRepository:
    def __init__(self, datasets: list[SeismicDataset] | None = None):
        self._datasets = {dataset.id: dataset for dataset in datasets or []}

    def get(self, dataset_id: int) -> SeismicDataset | None:
        return self._datasets.get(dataset_id)


class FakeJobRepository:
    """`get()` returns the exact same mutable Job instance stored by add(),
    so observing FAILED/COMPLETED/CANCELLED via get_job_status() only shows
    that the in-memory object was mutated -- it proves nothing about
    whether update() was ever called. `update_calls` records a snapshot
    (job_id, status, error_message) -- all immutable values -- at the
    moment each update() call happens, so tests can tell "the object
    changed" apart from "the repository was told to persist it".
    """

    def __init__(self) -> None:
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self.update_calls: list[tuple[int, JobStatus, str | None]] = []
        self.progress_calls: list[tuple[int, JobStatus, str | None, float]] = []

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
        self.update_calls.append((job.id, job.status, job.error_message))
        self.progress_calls.append(
            (job.id, job.status, job.error_message, job.progress)
        )

    def list(self, dataset_id=None, status=None):
        return list(self._jobs.values())


class FakeTraceReader:
    def __init__(self, traces: np.ndarray):
        self.traces = traces
        self.trace_count = len(traces)
        self.read_calls: list[tuple[int, int]] = []
        self.closed = False

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        self.read_calls.append((start, stop))
        return self.traces[start:stop]

    def close(self) -> None:
        self.closed = True


class FakeTraceWriter:
    """Test double distinguishing finalize() (mark output complete) from
    close() (release the resource) -- the two are never the same call.

    `job`, when given, lets a test observe the job's status at the moment
    finalize() runs (via `job_status_at_finalize`), to prove finalize()
    happens strictly before the job transitions to COMPLETED.
    """

    output_path = "/fake/output.h5"

    def __init__(self, job: Job | None = None) -> None:
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False
        self.closed = False
        self.calls: list[str] = []
        self._job = job
        self.job_status_at_finalize: JobStatus | None = None

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self.calls.append("finalize")
        self.finalized = True
        if self._job is not None:
            self.job_status_at_finalize = self._job.status

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True


class WriteChunkRaisingTraceWriter:
    """Writer whose write_chunk() fails, to prove mid-run I/O errors still
    clean up resources without ever treating the output as finalized.
    """

    output_path = "/fake/output.h5"

    def __init__(self) -> None:
        self.finalized = False
        self.closed = False

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        raise OSError("disk full")

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self.finalized = True

    def close(self) -> None:
        self.closed = True


class FinalizeRaisingTraceWriter:
    """Writer whose finalize() fails after every chunk was already written,
    to prove a failed finalize() never counts as success and still cleans
    up.
    """

    output_path = "/fake/output.h5"

    def __init__(self) -> None:
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False
        self.closed = False
        self.calls: list[str] = []

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self.calls.append("finalize")
        raise OSError("cannot flush output store")

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True


class CloseRaisingTraceReader:
    """Reader whose close() itself fails, to prove a cleanup failure on one
    resource must not block cleanup of the other, nor the job's terminal
    persistence.
    """

    def __init__(self, traces: np.ndarray):
        self.traces = traces
        self.trace_count = len(traces)
        self.read_calls: list[tuple[int, int]] = []
        self.close_attempted = False

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        self.read_calls.append((start, stop))
        return self.traces[start:stop]

    def close(self) -> None:
        self.close_attempted = True
        raise OSError("reader close failed")


class CloseRaisingTraceWriter:
    """Writer whose close() itself fails after a normal finalize(), to
    prove close() failing must not undo an already-valid job transition.
    """

    output_path = "/fake/output.h5"

    def __init__(self) -> None:
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False
        self.close_attempted = False

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self.finalized = True

    def close(self) -> None:
        self.close_attempted = True
        raise OSError("writer close failed")


class WriteChunkAndCloseRaisingTraceWriter:
    """Writer that fails during processing (write_chunk) and whose close()
    also fails, to prove the primary processing error is preserved as
    error_message even when the subsequent cleanup attempt fails too.
    """

    output_path = "/fake/output.h5"

    def __init__(self) -> None:
        self.finalized = False
        self.close_attempted = False

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        raise OSError("disk full")

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self.finalized = True

    def close(self) -> None:
        self.close_attempted = True
        raise OSError("writer close failed")


class FakeCancelToken:
    """`cancel_after` keeps the existing deterministic-by-call-count
    trigger used by earlier tests. `request_cancel()` is the half of the
    contract FilterJobService.cancel_job() uses to signal this same token
    instance from outside run_filter_job() -- once requested, is_cancelled()
    reports True from then on, regardless of `cancel_after`.
    """

    def __init__(self, cancel_after: int | None = None):
        self.calls = 0
        self._cancel_after = cancel_after
        self._requested = False

    def is_cancelled(self) -> bool:
        self.calls += 1
        if self._requested:
            return True
        if self._cancel_after is None:
            return False
        return self.calls > self._cancel_after

    def request_cancel(self) -> None:
        self._requested = True


class CancelRequestingTraceReader:
    """Reader that, from within read_chunk() for a chosen chunk, calls
    service.cancel_job(job_id) -- simulating a cancellation request
    arriving from another thread (e.g. the Qt GUI thread) while that
    chunk is mid-flight, deterministically and without sleep() or real
    threads. The chunk already being read must still return normally;
    only the *next* chunk boundary is expected to observe the request.
    """

    def __init__(
        self,
        traces: np.ndarray,
        service: FilterJobService,
        job_id: int,
        cancel_during_chunk: int,
    ):
        self.traces = traces
        self.trace_count = len(traces)
        self.read_calls: list[tuple[int, int]] = []
        self.closed = False
        self._service = service
        self._job_id = job_id
        self._cancel_during_chunk = cancel_during_chunk

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        self.read_calls.append((start, stop))
        if len(self.read_calls) == self._cancel_during_chunk:
            self._service.cancel_job(self._job_id)
        return self.traces[start:stop]

    def close(self) -> None:
        self.closed = True


class DoubleCancelRequestingTraceReader(CancelRequestingTraceReader):
    """Same as CancelRequestingTraceReader, but issues the cancellation
    request twice in a row, to prove repeated cancel_job() calls while the
    job is still RUNNING are idempotent.
    """

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        self.read_calls.append((start, stop))
        if len(self.read_calls) == self._cancel_during_chunk:
            self._service.cancel_job(self._job_id)
            self._service.cancel_job(self._job_id)
        return self.traces[start:stop]


@pytest.fixture
def dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=401,
        n_crosslines=720,
        n_traces=10,  # matches the fake readers below (physical count, not 401*720)
        n_samples=850,
        sample_rate_ms=4.0,  # nyquist_hz == 125.0
    )


def _build_service(
    dataset: SeismicDataset,
    reader: TraceReader,
    writer: TraceWriter,
    chunk_size: int = 4,
    jobs: FakeJobRepository | None = None,
) -> FilterJobService:
    return FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=jobs if jobs is not None else FakeJobRepository(),
        reader_factory=lambda ds: reader,
        writer_factory=lambda job, dataset: writer,
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize("parallel_workers", [0, -1, True])
def test_service_rejects_non_positive_parallel_worker_count(dataset, parallel_workers):
    with pytest.raises(ValueError, match="parallel_workers must be at least 1"):
        FilterJobService(
            datasets=FakeDatasetRepository([dataset]),
            jobs=FakeJobRepository(),
            parallel_workers=parallel_workers,
        )


def test_one_worker_preserves_sequential_filter_path(dataset, monkeypatch):
    traces = np.zeros((dataset.n_traces, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()

    def unexpected_pool(*args, **kwargs):
        raise AssertionError("parallel_workers=1 must not create a process pool")

    monkeypatch.setattr(
        "giecar_seismic.application.filter_jobs.ProcessPoolExecutor", unexpected_pool
    )
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda _dataset: reader,
        writer_factory=lambda _job, _dataset: writer,
        chunk_size=2,
        parallel_workers=1,
    )
    job = service.create_filter_job(dataset.id, 30.0, 4)

    finished = service.run_filter_job(job.id, lambda _pct: None, FakeCancelToken())

    assert finished.status is JobStatus.COMPLETED
    assert reader.read_calls == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]


def test_parallel_pool_is_reused_and_shutdown_after_subprocess_failure(
    dataset, monkeypatch
):
    traces = np.zeros((dataset.n_traces, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    pools = []

    class FakePool:
        def __init__(self, *, max_workers, mp_context):
            self.max_workers = max_workers
            self.start_method = mp_context.get_start_method()
            self.shutdown_calls = []
            pools.append(self)

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_calls.append((wait, cancel_futures))

    def fail_filter(*args, **kwargs):
        raise RuntimeError("subprocess filtering failed")

    monkeypatch.setattr(
        "giecar_seismic.application.filter_jobs.ProcessPoolExecutor", FakePool
    )
    monkeypatch.setattr(
        "giecar_seismic.application.filter_jobs.filter_chunk_with_executor",
        fail_filter,
    )
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda _dataset: reader,
        writer_factory=lambda _job, _dataset: writer,
        chunk_size=2,
        parallel_workers=3,
    )
    job = service.create_filter_job(dataset.id, 30.0, 4)

    result = service.run_filter_job(job.id, lambda _pct: None, FakeCancelToken())

    assert result.status is JobStatus.FAILED
    assert result.error_message == "subprocess filtering failed"
    assert writer.written == []
    assert writer.finalized is False
    assert len(pools) == 1
    assert pools[0].max_workers == 3
    assert pools[0].start_method == "spawn"
    assert pools[0].shutdown_calls == [(True, True)]


def test_parallel_pool_is_created_once_and_reused_for_all_chunks(dataset, monkeypatch):
    from concurrent.futures import Future

    traces = np.arange(dataset.n_traces * dataset.n_samples, dtype=np.float32).reshape(
        dataset.n_traces, dataset.n_samples
    )
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    pools = []

    class ImmediatePool:
        def __init__(self, *, max_workers, mp_context):
            self.submit_count = 0
            self.shutdown_count = 0
            pools.append(self)

        def submit(self, fn, *args):
            self.submit_count += 1
            future = Future()
            future.set_result(fn(*args))
            return future

        def shutdown(self, *, wait, cancel_futures):
            assert (wait, cancel_futures) == (True, True)
            self.shutdown_count += 1

    monkeypatch.setattr(
        "giecar_seismic.application.filter_jobs.ProcessPoolExecutor", ImmediatePool
    )
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda _dataset: reader,
        writer_factory=lambda _job, _dataset: writer,
        chunk_size=4,
        parallel_workers=3,
    )
    job = service.create_filter_job(dataset.id, 30.0, 4)

    result = service.run_filter_job(job.id, lambda _pct: None, FakeCancelToken())

    assert result.status is JobStatus.COMPLETED
    assert len(pools) == 1
    assert pools[0].submit_count == 8  # 3 + 3 + 2 non-empty partitions
    assert pools[0].shutdown_count == 1


def test_run_filter_job_completes_and_writes_all_chunks(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    created_writers: list[FakeTraceWriter] = []

    def writer_factory(job: Job, dataset: SeismicDataset) -> FakeTraceWriter:
        # capture `job` so the writer can record the job's status at the
        # instant finalize() runs -- see job_status_at_finalize below.
        writer = FakeTraceWriter(job=job)
        created_writers.append(writer)
        return writer

    jobs = FakeJobRepository()
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=jobs,
        reader_factory=lambda ds: reader,
        writer_factory=writer_factory,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    progresses: list[float] = []
    service.run_filter_job(
        job.id, progress_callback=progresses.append, cancel_token=FakeCancelToken()
    )

    writer = created_writers[0]
    completed = service.get_job_status(job.id)
    assert completed.status is JobStatus.COMPLETED
    assert writer.finalized is True
    assert reader.read_calls == [(0, 4), (4, 8), (8, 10)]
    assert sum(chunk.shape[0] for _, chunk in writer.written) == 10
    assert progresses == sorted(progresses)
    assert progresses[-1] == 100
    # reader is always closed, on every path.
    assert reader.closed is True
    # get_job_status() alone proves nothing about persistence here, since
    # FakeJobRepository.get() returns the very same mutable Job instance
    # that was mutated in place -- it would read COMPLETED even if
    # update() were never called. update_calls is the actual evidence that
    # JobRepository.update() was invoked, and with which status each time.
    statuses_persisted = [status for _, status, _ in jobs.update_calls]
    assert JobStatus.RUNNING in statuses_persisted
    assert statuses_persisted[-1] is JobStatus.COMPLETED
    # finalize() ("mark the output complete") and close() ("release the
    # resource") are distinct operations -- on success, both happen, in
    # that order: the writer is finalized, then released.
    assert writer.calls == ["finalize", "close"]
    assert writer.closed is True
    # finalize() must have run strictly before the job became COMPLETED.
    assert writer.job_status_at_finalize is JobStatus.RUNNING


def test_run_filter_job_stops_cooperatively_when_cancelled_mid_run(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
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
    # cancellation must still release both resources; finalize() is never
    # used to mean "cancelled" -- close() is the cleanup call here.
    assert reader.closed is True
    assert writer.closed is True
    # explicit evidence that update() persisted CANCELLED -- not just that
    # the shared in-memory Job object was mutated.
    statuses_persisted = [status for _, status, _ in jobs.update_calls]
    assert JobStatus.RUNNING in statuses_persisted
    assert statuses_persisted[-1] is JobStatus.CANCELLED


class RaisingTraceReader:
    trace_count = 10

    def __init__(self) -> None:
        self.closed = False

    def read_chunk(self, start: int, stop: int) -> np.ndarray:
        raise OSError("segyio read error")

    def close(self) -> None:
        self.closed = True


def test_run_filter_job_fails_the_job_when_reading_raises(dataset):
    writer = FakeTraceWriter()
    reader = RaisingTraceReader()
    jobs = FakeJobRepository()
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=jobs,
        reader_factory=lambda ds: reader,
        writer_factory=lambda job, dataset: writer,
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
    # reader was opened successfully before the failure in the chunk loop,
    # so it must still be closed; the writer was opened too and must be
    # released via close(), never finalize().
    assert reader.closed is True
    assert writer.closed is True
    # explicit evidence that update() persisted FAILED with the original
    # error -- not just that the shared in-memory Job object was mutated.
    assert (job.id, JobStatus.FAILED, "segyio read error") in jobs.update_calls


def test_run_filter_job_fails_and_cleans_up_when_write_chunk_raises(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = WriteChunkRaisingTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=4)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error_message == "disk full"
    assert writer.finalized is False
    assert reader.closed is True
    assert writer.closed is True


def test_run_filter_job_fails_when_reader_factory_raises_before_start_completes(
    dataset,
):
    writer_factory_calls: list[Job] = []

    def raising_reader_factory(ds: SeismicDataset) -> FakeTraceReader:
        raise OSError("cannot open segy file")

    def recording_writer_factory(job: Job, dataset: SeismicDataset) -> FakeTraceWriter:
        writer_factory_calls.append(job)
        return FakeTraceWriter()

    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=raising_reader_factory,
        writer_factory=recording_writer_factory,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.status is not JobStatus.RUNNING
    assert failed.error_message == "cannot open segy file"
    # the writer must never be created once opening the reader has failed
    assert writer_factory_calls == []


def test_run_filter_job_closes_reader_and_fails_when_writer_factory_raises(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)

    def raising_writer_factory(job: Job, dataset: SeismicDataset) -> FakeTraceWriter:
        raise OSError("cannot open output store")

    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda ds: reader,
        writer_factory=raising_writer_factory,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error_message == "cannot open output store"
    # the reader was already created and must not leak
    assert reader.closed is True


def test_run_filter_job_fails_when_finalize_raises_after_all_chunks_written(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FinalizeRaisingTraceWriter()
    service = _build_service(dataset, reader, writer, chunk_size=4)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    # all chunks were processed before finalize() was attempted
    assert reader.read_calls == [(0, 4), (4, 8), (8, 10)]
    assert sum(chunk.shape[0] for _, chunk in writer.written) == 10
    # a failed finalize() must never result in COMPLETED
    assert failed.status is JobStatus.FAILED
    assert failed.status is not JobStatus.COMPLETED
    assert failed.error_message == "cannot flush output store"
    # finalize() was attempted (and failed) -- it must never be treated as
    # having succeeded, so `finalized` stays False.
    assert writer.finalized is False
    # both resources must still be released even though finalize() failed.
    assert reader.closed is True
    assert writer.closed is True
    assert writer.calls == ["finalize", "close"]


def test_run_filter_job_completes_even_when_reader_close_raises(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = CloseRaisingTraceReader(traces)
    writer = FakeTraceWriter()
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    completed = service.get_job_status(job.id)
    # a cleanup failure must not leave the job stuck in RUNNING, and must
    # not prevent the already-decided terminal state from being persisted.
    assert completed.status is JobStatus.COMPLETED
    assert completed.status is not JobStatus.RUNNING
    assert reader.close_attempted is True
    # the writer's own cleanup must still be attempted even though the
    # reader's close() raised first -- cleanup is independent per resource.
    assert writer.finalized is True
    assert writer.closed is True
    # explicit evidence update() was actually called with COMPLETED --
    # get_job_status() alone can't distinguish this from update() never
    # having run, since the fake repository returns the same mutable Job.
    assert (job.id, JobStatus.COMPLETED, None) in jobs.update_calls


def test_run_filter_job_persists_completed_even_when_writer_close_raises(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = CloseRaisingTraceWriter()
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    completed = service.get_job_status(job.id)
    # writer.close() failing must not undo the already-valid COMPLETED
    # transition, and the reader must still have been closed.
    assert completed.status is JobStatus.COMPLETED
    assert writer.finalized is True
    assert writer.close_attempted is True
    assert reader.closed is True
    # explicit evidence that update() was still called (and with
    # COMPLETED) after writer.close() raised -- a cleanup failure must not
    # skip the final persistence attempt.
    assert (job.id, JobStatus.COMPLETED, None) in jobs.update_calls


def test_run_filter_job_stays_cancelled_even_when_writer_close_raises(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = CloseRaisingTraceWriter()
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    cancel_token = FakeCancelToken(cancel_after=1)
    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=cancel_token
    )

    cancelled = service.get_job_status(job.id)
    # writer.close() failing must not undo the already-valid CANCELLED
    # transition -- cancellation semantics are unchanged by this task.
    assert cancelled.status is JobStatus.CANCELLED
    assert writer.finalized is False
    assert writer.close_attempted is True
    assert reader.closed is True
    # explicit evidence that update() was still called (and with
    # CANCELLED) after writer.close() raised.
    assert (job.id, JobStatus.CANCELLED, None) in jobs.update_calls


def test_run_filter_job_preserves_primary_error_when_both_closes_also_fail(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = CloseRaisingTraceReader(traces)
    writer = WriteChunkAndCloseRaisingTraceWriter()
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.status is not JobStatus.RUNNING
    # the primary processing error must survive both cleanup failures --
    # a cleanup exception must never silently replace error_message.
    assert failed.error_message == "disk full"
    assert writer.finalized is False
    # both resources' close() must have been attempted despite both failing.
    assert reader.close_attempted is True
    assert writer.close_attempted is True
    # get_job_status() alone does NOT prove update() ran -- the fake
    # repository hands back the same mutated Job object regardless. The
    # real evidence that persistence of FAILED was attempted (even though
    # both close() calls raised) is this recorded update() call.
    assert (job.id, JobStatus.FAILED, "disk full") in jobs.update_calls


def test_run_filter_job_defers_a_cancellation_requested_mid_chunk(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    writer = FakeTraceWriter()
    reader_holder: list[CancelRequestingTraceReader] = []
    jobs = FakeJobRepository()
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=jobs,
        reader_factory=lambda ds: reader_holder[0],
        writer_factory=lambda job, dataset: writer,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    # 10 traces / chunk_size 4 -> 3 chunks: (0,4), (4,8), (8,10). Request
    # cancellation while the 2nd chunk (not the last) is being read.
    reader_holder.append(
        CancelRequestingTraceReader(traces, service, job.id, cancel_during_chunk=2)
    )

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    cancelled = service.get_job_status(job.id)
    reader = reader_holder[0]
    # the chunk that was already in flight when the request landed (chunk
    # 2) completed normally: it was read AND written. Only the 3rd chunk
    # -- which had not started yet -- was never begun.
    assert reader.read_calls == [(0, 4), (4, 8)]
    assert len(writer.written) == 2
    assert cancelled.status is JobStatus.CANCELLED
    assert writer.finalized is False
    assert reader.closed is True
    assert writer.closed is True
    assert (job.id, JobStatus.CANCELLED, None) in jobs.update_calls


def test_run_filter_job_observes_cancellation_requested_during_the_last_chunk(
    dataset,
):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    writer = FakeTraceWriter()
    reader_holder: list[CancelRequestingTraceReader] = []
    jobs = FakeJobRepository()
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=jobs,
        reader_factory=lambda ds: reader_holder[0],
        writer_factory=lambda job, dataset: writer,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    # 3 chunks total; request cancellation while reading the 3rd (last,
    # partial) chunk. There is no "next" chunk whose pre-loop check could
    # observe this -- the only place left to observe it is the boundary
    # between the loop finishing and writer.finalize() being called.
    reader_holder.append(
        CancelRequestingTraceReader(traces, service, job.id, cancel_during_chunk=3)
    )

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    cancelled = service.get_job_status(job.id)
    reader = reader_holder[0]
    # the last chunk had already been read and written before the request
    # was observed -- it is allowed to remain written, per the cooperative
    # (not abrupt) cancellation contract.
    assert reader.read_calls == [(0, 4), (4, 8), (8, 10)]
    assert len(writer.written) == 3
    # but the output must never be treated as a complete, valid result.
    assert writer.finalized is False
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.status is not JobStatus.COMPLETED
    assert reader.closed is True
    assert writer.closed is True
    assert (job.id, JobStatus.CANCELLED, None) in jobs.update_calls


def test_run_filter_job_treats_repeated_cancel_requests_idempotently(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    writer = FakeTraceWriter()
    reader_holder: list[DoubleCancelRequestingTraceReader] = []
    jobs = FakeJobRepository()
    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=jobs,
        reader_factory=lambda ds: reader_holder[0],
        writer_factory=lambda job, dataset: writer,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    reader_holder.append(
        DoubleCancelRequestingTraceReader(
            traces, service, job.id, cancel_during_chunk=1
        )
    )

    # two cancel_job() calls happen back to back while the job is still
    # RUNNING (see DoubleCancelRequestingTraceReader) -- neither may raise,
    # corrupt state, or force more than one terminal transition.
    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    cancelled = service.get_job_status(job.id)
    assert cancelled.status is JobStatus.CANCELLED
    # exactly one CANCELLED persistence, from the worker -- not two.
    cancelled_persist_count = sum(
        1 for _, status, _ in jobs.update_calls if status is JobStatus.CANCELLED
    )
    assert cancelled_persist_count == 1


# --- Point-of-no-return and duplicate-execution races -----------------------
#
# The tests below use real threading.Thread instances plus threading.Event
# for ordering. No sleep() anywhere: each side blocks on an Event until the
# other side signals it, so the interleaving under test is deterministic.


class BlockingLastChunkTraceWriter:
    """Writer whose write_chunk(), on the chunk numbered
    `block_after_write_count`, signals `about_to_check` and then blocks on
    `resume` before returning.

    Lets a test pause the worker at the exact instant after the last
    chunk has been written and before it reaches the atomic
    point-of-no-return check (which happens immediately afterwards, with
    no I/O in between).
    """

    output_path = "/fake/output.h5"

    def __init__(
        self,
        about_to_check: threading.Event,
        resume: threading.Event,
        block_after_write_count: int,
    ):
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False
        self.closed = False
        self._about_to_check = about_to_check
        self._resume = resume
        self._block_after_write_count = block_after_write_count

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))
        if len(self.written) == self._block_after_write_count:
            self._about_to_check.set()
            assert self._resume.wait(timeout=5), "test deadlocked: resume never set"

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self.finalized = True

    def close(self) -> None:
        self.closed = True


def test_cancel_accepted_immediately_before_the_point_of_no_return_yields_cancelled(
    dataset,
):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    about_to_check = threading.Event()
    resume = threading.Event()
    # 10 traces / chunk_size 4 -> 3 chunks; pause right after the 3rd
    # (last) chunk has been written.
    writer = BlockingLastChunkTraceWriter(
        about_to_check, resume, block_after_write_count=3
    )
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    cancel_token = CooperativeCancelToken()

    worker = threading.Thread(
        target=lambda: service.run_filter_job(
            job.id, progress_callback=lambda _pct: None, cancel_token=cancel_token
        )
    )
    worker.start()
    assert about_to_check.wait(timeout=5), "test deadlocked: worker never paused"

    # this request happens-before the worker's atomic point-of-no-return
    # check (the worker is still blocked in write_chunk), so it must be
    # accepted.
    service.cancel_job(job.id)

    resume.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    # an accepted request must result in CANCELLED -- never a silent
    # COMPLETED, even though the worker had already written every chunk.
    cancelled = service.get_job_status(job.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert writer.finalized is False
    assert (job.id, JobStatus.CANCELLED, None) in jobs.update_calls


class BlockingFinalizeTraceWriter:
    """Writer whose finalize() signals `finalize_started` -- proving the
    worker has already committed past the point of no return -- and then
    blocks on `resume` before actually completing.
    """

    output_path = "/fake/output.h5"

    def __init__(self, finalize_started: threading.Event, resume: threading.Event):
        self.written: list[tuple[int, np.ndarray]] = []
        self.finalized = False
        self.closed = False
        self._finalize_started = finalize_started
        self._resume = resume

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written.append((start, chunk))

    def checkpoint(self) -> None:
        pass  # In-memory writer has no pending disk buffers.

    def finalize(self) -> None:
        self._finalize_started.set()
        assert self._resume.wait(timeout=5), "test deadlocked: resume never set"
        self.finalized = True

    def close(self) -> None:
        self.closed = True


def test_cancel_rejected_after_the_point_of_no_return_has_been_crossed(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    finalize_started = threading.Event()
    resume = threading.Event()
    writer = BlockingFinalizeTraceWriter(finalize_started, resume)
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    cancel_token = CooperativeCancelToken()

    worker = threading.Thread(
        target=lambda: service.run_filter_job(
            job.id, progress_callback=lambda _pct: None, cancel_token=cancel_token
        )
    )
    worker.start()
    assert finalize_started.wait(timeout=5), "test deadlocked: finalize never started"

    # the worker has already committed to finalize() -- cancel_job() must
    # reject this explicitly rather than silently accepting a request it
    # can no longer honor.
    with pytest.raises(CancellationWindowClosedError):
        service.cancel_job(job.id)

    resume.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    completed = service.get_job_status(job.id)
    assert completed.status is JobStatus.COMPLETED
    assert writer.finalized is True
    assert (job.id, JobStatus.COMPLETED, None) in jobs.update_calls


def _build_service_with_blocking_reader_factory(
    dataset: SeismicDataset,
) -> tuple[
    FilterJobService, Job, threading.Event, threading.Event, FakeTraceWriter, list[int]
]:
    """Builds a service whose reader_factory blocks on `release` right
    after signalling `entered` -- i.e. strictly *after* run_filter_job()
    has already registered its token and flipped the job to RUNNING, but
    before any chunk has been read. Lets a test guarantee a first run has
    already claimed ownership before a second run is attempted.
    """
    entered = threading.Event()
    release = threading.Event()
    writer = FakeTraceWriter()
    reader_factory_call_count: list[int] = []

    def reader_factory(ds: SeismicDataset) -> FakeTraceReader:
        reader_factory_call_count.append(1)
        entered.set()
        assert release.wait(timeout=5), "test deadlocked: release never set"
        traces = np.zeros((10, ds.n_samples), dtype=np.float32)
        return FakeTraceReader(traces)

    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=reader_factory,
        writer_factory=lambda job, dataset: writer,
        chunk_size=4,
    )
    assert dataset.id is not None
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    return service, job, entered, release, writer, reader_factory_call_count


def test_run_filter_job_rejects_a_concurrent_second_execution_of_the_same_job(
    dataset,
):
    service, job, entered, release, writer, reader_factory_call_count = (
        _build_service_with_blocking_reader_factory(dataset)
    )
    token1 = CooperativeCancelToken()

    first_thread = threading.Thread(
        target=lambda: service.run_filter_job(
            job.id, progress_callback=lambda _pct: None, cancel_token=token1
        )
    )
    first_thread.start()
    # wait until the first run has already registered its token and
    # entered reader_factory() -- i.e. strictly after registration.
    assert entered.wait(timeout=5), "test deadlocked: first run never entered"

    second_run_errors: list[BaseException] = []

    def run_second() -> None:
        try:
            service.run_filter_job(
                job.id,
                progress_callback=lambda _pct: None,
                cancel_token=FakeCancelToken(),
            )
        except BaseException as exc:  # noqa: BLE001 -- captured for the assertion below
            second_run_errors.append(exc)

    second_thread = threading.Thread(target=run_second)
    second_thread.start()
    second_thread.join(timeout=5)
    assert not second_thread.is_alive()

    assert len(second_run_errors) == 1
    assert isinstance(second_run_errors[0], JobAlreadyRunningError)
    # the rejected duplicate must never have reached the reader factory --
    # it never created a reader/writer of its own.
    assert reader_factory_call_count == [1]

    # letting the first (legitimate) run finish proves it was never
    # disturbed by the rejected duplicate attempt.
    release.set()
    first_thread.join(timeout=5)
    assert not first_thread.is_alive()

    completed = service.get_job_status(job.id)
    assert completed.status is JobStatus.COMPLETED
    assert writer.finalized is True


def test_cancel_job_still_signals_the_original_run_after_a_duplicate_is_rejected(
    dataset,
):
    service, job, entered, release, writer, _reader_factory_call_count = (
        _build_service_with_blocking_reader_factory(dataset)
    )
    token1 = CooperativeCancelToken()

    first_thread = threading.Thread(
        target=lambda: service.run_filter_job(
            job.id, progress_callback=lambda _pct: None, cancel_token=token1
        )
    )
    first_thread.start()
    assert entered.wait(timeout=5), "test deadlocked: first run never entered"

    with pytest.raises(JobAlreadyRunningError):
        service.run_filter_job(
            job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
        )

    # cancel_job() must still reach the *first* run's token -- the
    # rejected duplicate attempt must not have overwritten or removed it.
    service.cancel_job(job.id)

    release.set()
    first_thread.join(timeout=5)
    assert not first_thread.is_alive()

    cancelled = service.get_job_status(job.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert writer.finalized is False


class DelayedSignalCancelToken:
    """Wraps a real CooperativeCancelToken but delays the moment
    request_cancel() actually flips it: it first signals `about_to_signal`
    and then blocks on `release_signal` before touching `inner` at all.

    Used to pin down the exact window between "cancel_job() decided to
    accept this request" and "the token is actually marked cancelled" --
    precisely the window a buggy implementation could leave open by
    releasing its registration lock before calling request_cancel().
    """

    def __init__(
        self,
        inner: CooperativeCancelToken,
        about_to_signal: threading.Event,
        release_signal: threading.Event,
    ):
        self._inner = inner
        self._about_to_signal = about_to_signal
        self._release_signal = release_signal

    def is_cancelled(self) -> bool:
        return self._inner.is_cancelled()

    def request_cancel(self) -> None:
        self._about_to_signal.set()
        assert self._release_signal.wait(timeout=5), (
            "test deadlocked: release_signal never set"
        )
        self._inner.request_cancel()


def test_cancel_job_accept_and_signal_are_atomic_with_the_point_of_no_return(dataset):
    """Regression test for the race between cancel_job() *accepting* a
    request and that request becoming *visible* to the worker.

    A cancel_job() call that finds `committed=False` has decided to
    accept the request. That decision and the request becoming visible to
    the worker's is_cancelled() check must be a single atomic step: it
    must be impossible for the worker to observe "not cancelled" and
    commit to finalize() after cancel_job() has already made that
    decision. This pins the exact interleaving a buggy implementation
    could allow by releasing its lock before actually signalling the
    token.
    """
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    about_to_check = threading.Event()
    resume_worker = threading.Event()
    # 10 traces / chunk_size 4 -> 3 chunks; pause right after the 3rd
    # (last) chunk has been written, immediately before the worker's
    # point-of-no-return check.
    writer = BlockingLastChunkTraceWriter(
        about_to_check, resume_worker, block_after_write_count=3
    )
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    about_to_signal = threading.Event()
    release_signal = threading.Event()
    cancel_token = DelayedSignalCancelToken(
        CooperativeCancelToken(), about_to_signal, release_signal
    )

    worker = threading.Thread(
        target=lambda: service.run_filter_job(
            job.id, progress_callback=lambda _pct: None, cancel_token=cancel_token
        ),
        daemon=True,
    )
    worker.start()
    assert about_to_check.wait(timeout=5), "test deadlocked: worker never paused"

    cancel_errors: list[BaseException] = []

    def call_cancel_job() -> None:
        try:
            service.cancel_job(job.id)
        except BaseException as exc:  # noqa: BLE001 -- captured for the assertion below
            cancel_errors.append(exc)

    canceller = threading.Thread(target=call_cancel_job, daemon=True)
    canceller.start()
    # wait until cancel_job() has already decided to accept the request
    # (registration.committed was False) and reached request_cancel() --
    # this is exactly the point where a buggy implementation would
    # already have released its registration lock.
    assert about_to_signal.wait(timeout=5), (
        "test deadlocked: cancel_job() never reached request_cancel()"
    )

    # let the worker try to cross the point of no return *before* the
    # already-accepted request has actually become visible.
    resume_worker.set()

    # A correct implementation keeps the registration lock held for the
    # whole accept-and-signal step, so the worker cannot possibly finish
    # here -- it can only be blocked trying to acquire that same lock.
    # Give it a generous, bounded window to prove that: the worker's
    # remaining work (an in-memory check, at most finalize()/complete() on
    # these fakes) is pure CPU-bound work with no I/O, so if it finishes
    # anyway within this window, request_cancel() must have run outside
    # the lock -- exactly the bug under test.
    worker.join(timeout=0.3)
    assert worker.is_alive(), (
        "worker crossed the point of no return before the already-"
        "accepted cancellation request became visible to it -- "
        "cancel_job() must hold its registration lock for the whole "
        "accept-and-signal step, not release it before calling "
        "request_cancel()"
    )

    # only now let the delayed signal actually happen.
    release_signal.set()

    worker.join(timeout=5)
    canceller.join(timeout=5)
    assert not worker.is_alive()
    assert not canceller.is_alive()
    assert cancel_errors == []

    # the request was accepted before the worker could commit -- it must
    # be honored: CANCELLED, never a silent COMPLETED.
    result = service.get_job_status(job.id)
    assert result.status is JobStatus.CANCELLED
    assert writer.finalized is False
    assert (job.id, JobStatus.CANCELLED, None) in jobs.update_calls


# --- Progress is persisted per chunk from the reader's physical count -----


def test_run_filter_job_persists_incremental_progress_from_physical_trace_count(
    dataset,
):
    # 10 physical traces / chunk_size 4 -> chunk boundaries at 4, 8, 10.
    # The dataset's grid says 401 x 720 = 288720 -- progress must come from
    # reader.trace_count (10, the physical count), never from the grid.
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    persisted_progress = [progress for (_, status, _, progress) in jobs.progress_calls]
    assert persisted_progress[-1] == 100
    assert persisted_progress == sorted(persisted_progress)
    running_progress = [
        progress
        for (_, status, _, progress) in jobs.progress_calls
        if status is JobStatus.RUNNING
    ]
    # start() persisted at 0, then output_path registration at 0 (once the
    # writer exists), then one update per chunk: 4/10, 8/10, 10/10.
    assert running_progress == [0, 0, 40.0, 80.0, 100.0]
    completed = service.get_job_status(job.id)
    assert completed.progress == 100
    assert completed.output_path == "/fake/output.h5"


def test_run_filter_job_cancellation_keeps_the_last_progress_reached(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    # cancelled before chunk 2: exactly one chunk (4/10) was processed.
    service.run_filter_job(
        job.id,
        progress_callback=lambda _pct: None,
        cancel_token=FakeCancelToken(cancel_after=1),
    )

    cancelled = service.get_job_status(job.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.progress == 40.0
    assert cancelled.finished_at is not None
    assert (job.id, JobStatus.CANCELLED, None, 40.0) in jobs.progress_calls


# --- output_path registration and physical trace-count validation ---------


def test_run_filter_job_fails_before_creating_a_writer_on_trace_count_mismatch(
    dataset,
):
    # dataset fixture says n_traces=10, but the file the reader opens now
    # has 12 -- e.g. it was replaced after import. Progress and the HDF5
    # size are derived from that count, so the job must fail clearly,
    # before any output file is created.
    traces = np.zeros((12, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer_factory_calls: list[Job] = []
    jobs = FakeJobRepository()

    def recording_writer_factory(job: Job, ds: SeismicDataset) -> FakeTraceWriter:
        writer_factory_calls.append(job)
        return FakeTraceWriter()

    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=jobs,
        reader_factory=lambda ds: reader,
        writer_factory=recording_writer_factory,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    assert failed.status is JobStatus.FAILED
    assert "12 traces" in (failed.error_message or "")
    assert "10" in (failed.error_message or "")
    assert failed.output_path is None
    assert writer_factory_calls == []
    assert reader.read_calls == []
    assert reader.closed is True


def test_run_filter_job_records_output_path_before_processing_any_chunk(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)
    writer = FakeTraceWriter()
    writer.output_path = "/out/job-1.h5"
    jobs = FakeJobRepository()
    service = _build_service(dataset, reader, writer, chunk_size=4, jobs=jobs)
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    assert service.get_job_status(job.id).output_path == "/out/job-1.h5"
    # the update that carried output_path happened while RUNNING at
    # progress 0 -- i.e. before the first chunk was written.
    assert (job.id, JobStatus.RUNNING, None, 0.0) in jobs.progress_calls


def test_run_filter_job_leaves_output_path_none_when_writer_factory_raises(dataset):
    traces = np.zeros((10, dataset.n_samples), dtype=np.float32)
    reader = FakeTraceReader(traces)

    def raising_writer_factory(job: Job, ds: SeismicDataset) -> FakeTraceWriter:
        raise OSError("cannot create output file")

    service = FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
        reader_factory=lambda ds: reader,
        writer_factory=raising_writer_factory,
        chunk_size=4,
    )
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    service.run_filter_job(
        job.id, progress_callback=lambda _pct: None, cancel_token=FakeCancelToken()
    )

    failed = service.get_job_status(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.output_path is None
