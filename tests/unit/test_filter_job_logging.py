import numpy as np
import pytest

from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.application.job_logging import JobLogLevel
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import FilterType, JobStatus
from tests.unit.test_run_filter_job import (
    FakeCancelToken,
    FakeDatasetRepository,
    FakeJobRepository,
    FakeTraceReader,
    FakeTraceWriter,
    WriteChunkRaisingTraceWriter,
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


class RecordingJobExecutionLogger:
    def __init__(self) -> None:
        self.records: list[tuple[int, JobLogLevel, str, str]] = []
        self.on_log = None

    def log(self, job_id, level, event, message):
        self.records.append((job_id, level, event, message))
        if self.on_log is not None:
            self.on_log(event)

    @property
    def events(self):
        return [record[2] for record in self.records]


def build_service(dataset, reader, writer, logger, *, chunk_size=2, resume=None):
    return FilterJobService(
        FakeDatasetRepository([dataset]),
        FakeJobRepository(),
        reader_factory=lambda _dataset: reader,
        writer_factory=lambda _job, _dataset: writer,
        resume_writer_factory=resume,
        chunk_size=chunk_size,
        execution_logger=logger,
    )


@pytest.mark.parametrize("chunk_size", [1, 5])
def test_success_logs_only_lifecycle_events_independent_of_chunk_count(
    dataset, chunk_size
):
    logger = RecordingJobExecutionLogger()
    writer = FakeTraceWriter()
    service = build_service(
        dataset,
        FakeTraceReader(np.zeros((dataset.n_traces, dataset.n_samples))),
        writer,
        logger,
        chunk_size=chunk_size,
    )
    job = service.create_filter_job(dataset.id, 30, 4)
    logger.on_log = lambda event: (
        pytest.fail("completion logged before finalize")
        if event == "JOB_COMPLETED" and not writer.finalized
        else None
    )

    result = service.run_filter_job(job.id, lambda _: None, FakeCancelToken())

    assert result.status is JobStatus.COMPLETED
    assert logger.events == ["JOB_CREATED", "RUN_STARTED", "JOB_COMPLETED"]


def test_failure_logs_failed_with_short_error(dataset):
    logger = RecordingJobExecutionLogger()
    service = build_service(
        dataset,
        FakeTraceReader(np.zeros((dataset.n_traces, dataset.n_samples))),
        WriteChunkRaisingTraceWriter(),
        logger,
    )
    job = service.create_filter_job(dataset.id, 30, 4)

    result = service.run_filter_job(job.id, lambda _: None, FakeCancelToken())

    assert result.status is JobStatus.FAILED
    assert logger.events == ["JOB_CREATED", "RUN_STARTED", "JOB_FAILED"]
    assert "disk full" in logger.records[-1][3]
    assert logger.records[-1][1] is JobLogLevel.ERROR


def test_accepted_cancel_request_and_effective_cancel_are_separate_events(dataset):
    logger = RecordingJobExecutionLogger()
    token = CooperativeCancelToken()
    service = build_service(
        dataset,
        FakeTraceReader(np.zeros((dataset.n_traces, dataset.n_samples))),
        FakeTraceWriter(),
        logger,
        chunk_size=2,
    )
    job = service.create_filter_job(dataset.id, 30, 4)

    def cancel_after_checkpoint(_percent):
        service.cancel_job(job.id)

    result = service.run_filter_job(job.id, cancel_after_checkpoint, token)

    assert result.status is JobStatus.CANCELLED
    assert logger.events == [
        "JOB_CREATED",
        "RUN_STARTED",
        "CANCEL_REQUESTED",
        "JOB_CANCELLED",
    ]
    assert logger.records[-2][1] is JobLogLevel.WARNING
    assert logger.records[-1][1] is JobLogLevel.WARNING


def test_logging_failure_never_changes_successful_job(dataset):
    class BrokenLogger:
        def log(self, job_id, level, event, message):
            raise OSError("logs are read-only")

    writer = FakeTraceWriter()
    service = build_service(
        dataset,
        FakeTraceReader(np.zeros((dataset.n_traces, dataset.n_samples))),
        writer,
        BrokenLogger(),
    )
    job = service.create_filter_job(dataset.id, 30, 4)

    result = service.run_filter_job(job.id, lambda _: None, FakeCancelToken())

    assert result.status is JobStatus.COMPLETED
    assert writer.finalized


# --- resume -------------------------------------------------------------------


class FakeResumeWriter(FakeTraceWriter):
    """In-memory writer that also answers the resume-side questions."""

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = output_path
        self.written: dict[int, np.ndarray] = {}

    def write_chunk(self, start: int, chunk: np.ndarray) -> None:
        self.written[start] = chunk

    @property
    def written_trace_count(self) -> int:
        return sum(len(c) for c in self.written.values())

    @property
    def is_complete(self) -> bool:
        return self.finalized


class FakeResumableReader(FakeTraceReader):
    def __init__(self, traces, sample_rate_ms):
        super().__init__(traces)
        self.sample_count = traces.shape[1]
        self.sample_rate_ms = sample_rate_ms


def _resumable_world(tmp_path, dataset):
    from dataclasses import replace

    from giecar_seismic.application.dataset_import import read_source_fingerprint

    source = tmp_path / "survey.segy"
    source.write_bytes(b"segy")
    dataset = replace(
        dataset,
        source_path=str(source),
        source_fingerprint=read_source_fingerprint(source),
    )
    logger = RecordingJobExecutionLogger()
    writer = FakeResumeWriter("/out/job.h5")
    traces = np.zeros((dataset.n_traces, dataset.n_samples))
    service = build_service(
        dataset,
        FakeResumableReader(traces, dataset.sample_rate_ms),
        writer,
        logger,
        chunk_size=2,
        resume=lambda _job, _dataset: writer,
    )
    return service, logger, writer


def _cancel_once(service, job_id, token):
    fired = {"done": False}

    def callback(_pct):
        if not fired["done"]:
            fired["done"] = True
            service.cancel_job(job_id)

    return callback


def test_resume_logs_resume_started_with_trace_and_then_completion(tmp_path, dataset):
    service, logger, _ = _resumable_world(tmp_path, dataset)
    job = service.create_filter_job(dataset.id, 30, 4)
    token = CooperativeCancelToken()
    job = service.run_filter_job(job.id, _cancel_once(service, job.id, token), token)
    assert job.status is JobStatus.CANCELLED and job.processed_traces == 2

    done = service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())

    assert done.status is JobStatus.COMPLETED
    assert logger.events == [
        "JOB_CREATED",
        "RUN_STARTED",
        "CANCEL_REQUESTED",
        "JOB_CANCELLED",
        "RESUME_STARTED",
        "JOB_COMPLETED",
    ]
    assert "2" in logger.records[4][3]  # resumes from trace 2
    assert all(record[0] == job.id for record in logger.records)


def test_cancel_resume_cancel_resume_complete_keeps_one_trajectory(tmp_path, dataset):
    service, logger, _ = _resumable_world(tmp_path, dataset)
    job = service.create_filter_job(dataset.id, 30, 4)
    token = CooperativeCancelToken()
    service.run_filter_job(job.id, _cancel_once(service, job.id, token), token)
    token = CooperativeCancelToken()
    service.resume_filter_job(job.id, _cancel_once(service, job.id, token), token)
    done = service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())

    assert done.status is JobStatus.COMPLETED and done.resume_count == 2
    assert logger.events == [
        "JOB_CREATED",
        "RUN_STARTED",
        "CANCEL_REQUESTED",
        "JOB_CANCELLED",
        "RESUME_STARTED",
        "CANCEL_REQUESTED",
        "JOB_CANCELLED",
        "RESUME_STARTED",
        "JOB_COMPLETED",
    ]


def test_rejected_cancel_requests_are_not_logged(dataset):
    from giecar_seismic.domain.job import InvalidTransitionError

    logger = RecordingJobExecutionLogger()
    service = build_service(
        dataset,
        FakeTraceReader(np.zeros((dataset.n_traces, dataset.n_samples))),
        FakeTraceWriter(),
        logger,
    )
    job = service.create_filter_job(dataset.id, 30, 4)

    with pytest.raises(InvalidTransitionError):
        service.cancel_job(job.id)  # CREATED, not RUNNING

    assert logger.events == ["JOB_CREATED"]


def test_run_started_message_carries_the_audit_parameters(dataset):
    logger = RecordingJobExecutionLogger()
    service = build_service(
        dataset,
        FakeTraceReader(np.zeros((dataset.n_traces, dataset.n_samples))),
        FakeTraceWriter(),
        logger,
        chunk_size=3,
    )
    job = service.create_filter_job(
        dataset.id, 10, 6, filter_type=FilterType.BAND_PASS, upper_cutoff_hz=40
    )
    service.run_filter_job(job.id, lambda _: None, FakeCancelToken())

    created = logger.records[0][3]
    started = logger.records[1][3]
    assert (
        "Band-pass" in created
        and "10" in created
        and "40" in created
        and "6" in created
    )
    assert (
        f"dataset {dataset.id}" in started.lower()
        or f"dataset_id={dataset.id}" in started
    )
    assert "chunk" in started.lower() and "3" in started
    completed = logger.records[2][3]
    assert f"{dataset.n_traces}" in completed and "/fake/output.h5" in completed
