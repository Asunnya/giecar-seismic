"""H1: a job left RUNNING by a crash must become CANCELLED on startup so the
validated HDF5 checkpoint is reachable through Resume."""

import threading

import numpy as np
import pytest

from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.application.job_logging import JobLogLevel
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus
from tests.unit.test_filter_job_logging import RecordingJobExecutionLogger
from tests.unit.test_run_filter_job import (
    FakeDatasetRepository,
    FakeJobRepository,
    FakeTraceReader,
    FakeTraceWriter,
)


@pytest.fixture
def dataset():
    return SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=2,
        n_crosslines=5,
        n_traces=10,
        n_samples=64,
        sample_rate_ms=4.0,
    )


class BlockingReader(FakeTraceReader):
    def __init__(self, traces, entered, release):
        super().__init__(traces)
        self._entered, self._release = entered, release

    def read_chunk(self, start, stop):
        self._entered.set()
        assert self._release.wait(timeout=5), "test deadlocked"
        return super().read_chunk(start, stop)


def _service(dataset, jobs, logger, reader=None):
    traces = np.zeros((dataset.n_traces, dataset.n_samples))
    return FilterJobService(
        FakeDatasetRepository([dataset]),
        jobs,
        reader_factory=lambda _d: reader or FakeTraceReader(traces),
        writer_factory=lambda _j, _d: FakeTraceWriter(),
        chunk_size=2,
        execution_logger=logger,
    )


def _interrupted(jobs, dataset, processed=4):
    job = Job(dataset_id=dataset.id, cutoff_hz=30, order=4, status=JobStatus.RUNNING)
    job.processed_traces = processed
    job.progress = 40.0
    job.output_path = "/out/job.h5"
    return jobs.add(job)


def test_running_job_without_execution_becomes_cancelled_and_is_logged(dataset):
    jobs, logger = FakeJobRepository(), RecordingJobExecutionLogger()
    interrupted = _interrupted(jobs, dataset)
    untouched = jobs.add(Job(dataset_id=dataset.id, cutoff_hz=30, order=4))
    done = Job(dataset_id=dataset.id, cutoff_hz=30, order=4, status=JobStatus.COMPLETED)
    done = jobs.add(done)
    service = _service(dataset, jobs, logger)

    recovered = service.recover_interrupted_jobs()

    assert [j.id for j in recovered] == [interrupted.id]
    stored = jobs.get(interrupted.id)
    assert stored.status is JobStatus.CANCELLED
    assert stored.finished_at is not None
    assert stored.processed_traces == 4  # the checkpoint is never touched
    assert stored.output_path == "/out/job.h5"
    assert (interrupted.id, JobStatus.CANCELLED, None) in jobs.update_calls
    assert jobs.get(untouched.id).status is JobStatus.CREATED
    assert jobs.get(done.id).status is JobStatus.COMPLETED
    assert logger.records == [
        (
            interrupted.id,
            JobLogLevel.WARNING,
            "JOB_INTERRUPTED",
            logger.records[0][3],
        )
    ]
    assert "4" in logger.records[0][3]


def test_recovery_is_idempotent_and_quiet_when_nothing_is_interrupted(dataset):
    jobs, logger = FakeJobRepository(), RecordingJobExecutionLogger()
    _interrupted(jobs, dataset)
    service = _service(dataset, jobs, logger)

    assert len(service.recover_interrupted_jobs()) == 1
    assert service.recover_interrupted_jobs() == []
    assert len(logger.records) == 1


def test_a_live_execution_is_never_mistaken_for_an_interrupted_one(dataset):
    jobs, logger = FakeJobRepository(), RecordingJobExecutionLogger()
    entered, release = threading.Event(), threading.Event()
    reader = BlockingReader(
        np.zeros((dataset.n_traces, dataset.n_samples)), entered, release
    )
    service = _service(dataset, jobs, logger, reader=reader)
    job = service.create_filter_job(dataset.id, 30, 4)
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(
            job=service.run_filter_job(job.id, lambda _: None, CooperativeCancelToken())
        )
    )
    worker.start()
    assert entered.wait(timeout=5)  # provably RUNNING with a registered execution

    assert service.recover_interrupted_jobs() == []
    assert jobs.get(job.id).status is JobStatus.RUNNING

    release.set()
    worker.join(timeout=5)
    assert result["job"].status is JobStatus.COMPLETED
    assert "JOB_INTERRUPTED" not in logger.events


def test_recovered_job_can_be_resumed_like_a_cancelled_one(tmp_path, dataset):
    from dataclasses import replace

    from giecar_seismic.application.dataset_import import read_source_fingerprint
    from tests.unit.test_filter_job_logging import (
        FakeResumableReader,
        FakeResumeWriter,
    )

    source = tmp_path / "survey.segy"
    source.write_bytes(b"segy")
    dataset = replace(
        dataset,
        source_path=str(source),
        source_fingerprint=read_source_fingerprint(source),
    )
    jobs, logger = FakeJobRepository(), RecordingJobExecutionLogger()
    writer = FakeResumeWriter("/out/job.h5")
    traces = np.zeros((dataset.n_traces, dataset.n_samples))
    writer.written[0] = traces[:4]  # the durable prefix of the crashed run
    reader = FakeResumableReader(traces, dataset.sample_rate_ms)
    service = FilterJobService(
        FakeDatasetRepository([dataset]),
        jobs,
        reader_factory=lambda _d: reader,
        writer_factory=lambda _j, _d: writer,
        resume_writer_factory=lambda _j, _d: writer,
        chunk_size=2,
        execution_logger=logger,
    )
    interrupted = _interrupted(jobs, dataset, processed=4)

    service.recover_interrupted_jobs()
    done = service.resume_filter_job(
        interrupted.id, lambda _: None, CooperativeCancelToken()
    )

    assert done.status is JobStatus.COMPLETED
    assert sorted(writer.written) == [0, 4, 6, 8]  # only the remainder was processed
    assert logger.events == ["JOB_INTERRUPTED", "RESUME_STARTED", "JOB_COMPLETED"]
