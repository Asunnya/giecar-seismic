"""Real stores and irregular SEG-Y: resume survives closing the application."""

import h5py
import numpy as np
import pytest
from test_end_to_end_filter_pipeline import FOOTPRINTS, compose, write_segy

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter
from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.domain.job import JobStatus
from giecar_seismic.infrastructure.database.engine import make_session_factory
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
)
from giecar_seismic.infrastructure.segy.reader import open_dataset_reader
from giecar_seismic.infrastructure.storage.hdf5_writer import open_hdf5_resume_writer


def setup_cancelled(tmp_path, chunk_size=2):
    source = tmp_path / "survey.segy"
    amplitudes = write_segy(source, FOOTPRINTS[0], 64, 4000)
    engine, importer, service = compose(
        tmp_path / "jobs.sqlite", tmp_path / "outputs", chunk_size
    )
    dataset = importer(str(source), "irregular")
    job = service.create_filter_job(dataset.id, 30, 4)
    token = CooperativeCancelToken()
    job = service.run_filter_job(job.id, lambda _: token.request_cancel(), token)
    return engine, dataset, job, amplitudes


def resumed_service(engine, reads, chunk_size=2):
    def reader_factory(dataset):
        reader = open_dataset_reader(dataset)
        original = reader.read_chunk

        def read(start, stop):
            reads.append((start, stop))
            return original(start, stop)

        reader.read_chunk = read
        return reader

    sessions = make_session_factory(engine)
    jobs = SqlAlchemyJobRepository(sessions)
    service = FilterJobService(
        SqlAlchemyDatasetRepository(sessions),
        jobs,
        reader_factory=reader_factory,
        resume_writer_factory=open_hdf5_resume_writer,
        chunk_size=chunk_size,
    )
    return service, jobs


def test_resume_e2e_reopens_stores_and_cancels_twice_without_reprocessing(tmp_path):
    engine, dataset, job, amplitudes = setup_cancelled(tmp_path)
    assert job.processed_traces == 2
    created, started, output = job.created_at, job.started_at, job.output_path
    engine.dispose()
    # A fresh engine and service, no in-memory execution state carried over.
    from giecar_seismic.infrastructure.database.engine import create_sqlite_engine

    engine = create_sqlite_engine(tmp_path / "jobs.sqlite")
    reads = []
    service, jobs = resumed_service(engine, reads)
    token = CooperativeCancelToken()

    def cancel_next_checkpoint(pct):
        if pct > job.progress:
            token.request_cancel()

    again = service.resume_filter_job(job.id, cancel_next_checkpoint, token)
    assert again.status is JobStatus.CANCELLED
    assert again.processed_traces == 4
    assert reads == [(2, 4)]
    engine.dispose()
    engine = create_sqlite_engine(tmp_path / "jobs.sqlite")
    service, jobs = resumed_service(engine, reads)
    progress = []
    done = service.resume_filter_job(job.id, progress.append, CooperativeCancelToken())
    assert done.status is JobStatus.COMPLETED
    assert done.processed_traces == dataset.n_traces
    assert done.resume_count == 2
    assert (done.created_at, done.started_at, done.output_path) == (
        created,
        started,
        output,
    )
    assert reads == [(2, 4), (4, 6), (6, 8)]
    assert progress == [50, 75, 100]
    assert jobs.get(job.id) == done
    # Independent uninterrupted pipeline, not just the scientific function.
    reference_engine, _, reference_service = compose(
        tmp_path / "jobs.sqlite", tmp_path / "outputs", 2
    )
    reference = reference_service.create_filter_job(dataset.id, 30, 4)
    reference = reference_service.run_filter_job(
        reference.id, lambda _: None, CooperativeCancelToken()
    )
    assert reference.status is JobStatus.COMPLETED
    with (
        h5py.File(reference.output_path, "r") as full,
        h5py.File(output, "r") as resumed,
    ):
        np.testing.assert_array_equal(resumed["traces"][:], full["traces"][:])
    reference_engine.dispose()
    with h5py.File(output, "r") as file:
        assert file.attrs["complete"]
        np.testing.assert_allclose(
            file["traces"][:],
            apply_lowpass_filter(amplitudes, 30, 4, 4).astype(np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
    engine.dispose()


@pytest.mark.parametrize("sql_count", [0, 2, 3])
def test_checkpoint_authority(tmp_path, sql_count):
    engine, _, job, _ = setup_cancelled(tmp_path)
    reads = []
    service, jobs = resumed_service(engine, reads)
    job.processed_traces = sql_count
    job.progress = 100 * sql_count / 8
    jobs.update(job)
    if sql_count > 2:
        from giecar_seismic.application.filter_jobs import CheckpointInconsistentError

        with pytest.raises(CheckpointInconsistentError):
            service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
        assert jobs.get(job.id).status is JobStatus.CANCELLED
        assert reads == []
    else:
        progress = []
        done = service.resume_filter_job(
            job.id, progress.append, CooperativeCancelToken()
        )
        assert done.status is JobStatus.COMPLETED
        assert reads[0] == (2, 4)
        assert progress[0] == 25
    engine.dispose()


@pytest.mark.parametrize("finalized", [False, True])
def test_full_checkpoint_crash_window_finalizes_without_filtering(tmp_path, finalized):
    engine, dataset, job, _ = setup_cancelled(tmp_path, chunk_size=8)
    if finalized:
        with h5py.File(job.output_path, "r+") as file:
            file.attrs["complete"] = True
    reads = []
    service, _ = resumed_service(engine, reads)
    done = service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    assert done.status is JobStatus.COMPLETED
    assert done.processed_traces == dataset.n_traces
    assert reads == []
    with h5py.File(job.output_path, "r") as file:
        assert file.attrs["complete"]
    engine.dispose()


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "no_path",
        "shape",
        "complete_partial",
        "source_changed",
        "unknown_source",
        "sample_count",
        "sample_interval",
        "trace_count",
    ],
)
def test_resume_preflight_rejects_unsafe_inputs_without_amplitude_reads(
    tmp_path, problem
):
    from dataclasses import replace

    from giecar_seismic.application.filter_jobs import (
        CheckpointInconsistentError,
        SourceChangedError,
        TraceCountMismatchError,
    )

    engine, dataset, job, _ = setup_cancelled(tmp_path)
    reads = []
    service, jobs = resumed_service(engine, reads)
    expected_error = CheckpointInconsistentError
    if problem == "missing":
        from pathlib import Path

        Path(job.output_path).unlink()
    elif problem == "no_path":
        job.output_path = None
        jobs.update(job)
    elif problem in ("shape", "complete_partial"):
        with h5py.File(job.output_path, "r+") as file:
            if problem == "shape":
                file["traces"].resize(3, axis=0)
            else:
                file.attrs["complete"] = True
    else:
        expected_error = SourceChangedError
        if problem == "source_changed":
            import os

            os.utime(dataset.source_path, ns=(1, 1))
        else:
            fields = {
                "unknown_source": {"source_fingerprint": None},
                "sample_count": {"n_samples": 65},
                "sample_interval": {"sample_rate_ms": 2},
                "trace_count": {"n_traces": 9},
            }
            changed = replace(dataset, **fields[problem])
            service._datasets.get = lambda _: changed
            if problem == "trace_count":
                expected_error = TraceCountMismatchError
    with pytest.raises(expected_error):
        service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    stored = jobs.get(job.id)
    assert stored.status is JobStatus.CANCELLED
    assert stored.resume_count == 0
    assert stored.processed_traces == 2
    assert reads == []
    assert service._cancel_tokens == {}
    engine.dispose()


@pytest.mark.parametrize(
    "status",
    [JobStatus.CREATED, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED],
)
def test_service_resume_rejects_other_statuses(tmp_path, status):
    from giecar_seismic.domain.job import InvalidTransitionError

    engine, _, job, _ = setup_cancelled(tmp_path)
    reads = []
    service, jobs = resumed_service(engine, reads)
    job.status = status
    jobs.update(job)
    with pytest.raises(InvalidTransitionError):
        service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    assert reads == []
    assert jobs.get(job.id).status is status
    engine.dispose()


def test_concurrent_resume_and_run_are_rejected_and_cancel_remains_registered(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from giecar_seismic.application.filter_jobs import JobAlreadyRunningError
    from giecar_seismic.infrastructure.storage.hdf5_writer import (
        make_hdf5_writer_factory,
    )

    engine, _, job, _ = setup_cancelled(tmp_path)
    service, _ = resumed_service(engine, [])
    service._writer_factory = make_hdf5_writer_factory(tmp_path / "outputs")
    entered, release = Event(), Event()

    def block_after_reconcile(_):
        entered.set()
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            service.resume_filter_job,
            job.id,
            block_after_reconcile,
            CooperativeCancelToken(),
        )
        try:
            assert entered.wait(5)
            for operation in (service.resume_filter_job, service.run_filter_job):
                with pytest.raises(JobAlreadyRunningError):
                    operation(job.id, lambda _: None, CooperativeCancelToken())
            service.cancel_job(job.id)
        finally:
            release.set()
        cancelled = future.result(timeout=5)
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.processed_traces == 2
    assert cancelled.resume_count == 1
    engine.dispose()


def test_sqlite_never_advances_before_hdf5_checkpoint_succeeds(tmp_path, monkeypatch):
    from giecar_seismic.infrastructure.storage.hdf5_writer import Hdf5TraceWriter

    engine, _, job, _ = setup_cancelled(tmp_path)
    service, jobs = resumed_service(engine, [])
    original_checkpoint = Hdf5TraceWriter.checkpoint
    events = []

    def checkpoint(writer):
        assert jobs.get(job.id).processed_traces == 2
        original_checkpoint(writer)
        events.append("hdf5")
        raise OSError("checkpoint acknowledgement failed")

    monkeypatch.setattr(Hdf5TraceWriter, "checkpoint", checkpoint)
    result = service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    assert result.status is JobStatus.FAILED
    assert jobs.get(job.id).processed_traces == 2
    assert events == ["hdf5"]
    engine.dispose()


def test_checkpoint_order_and_crash_between_hdf5_and_sqlite(tmp_path, monkeypatch):
    engine, _, job, _ = setup_cancelled(tmp_path)
    service, jobs = resumed_service(engine, [])

    # Simulate a process interruption at the exact boundary. The persisted
    # status is explicitly CANCELLED before retry: resume never claims RUNNING.
    class Crash(BaseException):
        pass

    original_update = jobs.update

    def update(value):
        if value.processed_traces > 2:
            raise Crash()
        original_update(value)

    monkeypatch.setattr(jobs, "update", update)
    with pytest.raises(Crash):
        service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    with h5py.File(job.output_path, "r") as file:
        assert file.attrs["written_trace_count"] == 4
    stored = jobs.get(job.id)
    assert stored.processed_traces == 2
    stored.cancel()
    original_update(stored)
    monkeypatch.setattr(jobs, "update", original_update)
    reads = []
    service, jobs = resumed_service(engine, reads)
    done = service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    assert done.status is JobStatus.COMPLETED
    assert reads == [(4, 6), (6, 8)]
    engine.dispose()


def test_resume_preserves_checkpoint_commit_callback_order(tmp_path, monkeypatch):
    from giecar_seismic.application import filter_jobs
    from giecar_seismic.infrastructure.storage.hdf5_writer import Hdf5TraceWriter

    engine, _, job, _ = setup_cancelled(tmp_path)
    service, jobs = resumed_service(engine, [])
    events = []
    factory = service._reader_factory

    def reader_factory(dataset):
        reader = factory(dataset)
        original = reader.read_chunk

        def read(start, stop):
            events.append(("read", stop))
            return original(start, stop)

        reader.read_chunk = read
        return reader

    service._reader_factory = reader_factory
    original_filter = filter_jobs.apply_butterworth_filter

    def filter_chunk(*args, **kwargs):
        events.append(("filter", len(args[0])))
        return original_filter(*args, **kwargs)

    monkeypatch.setattr(filter_jobs, "apply_butterworth_filter", filter_chunk)
    original_write = Hdf5TraceWriter.write_chunk
    original_checkpoint = Hdf5TraceWriter.checkpoint

    def write(writer, start, chunk):
        events.append(("write", start + len(chunk)))
        original_write(writer, start, chunk)

    def checkpoint(writer):
        original_checkpoint(writer)
        with h5py.File(writer.output_path, "r") as file:
            assert file.attrs["written_trace_count"] == writer.written_trace_count
        events.append(("checkpoint", writer.written_trace_count))

    monkeypatch.setattr(Hdf5TraceWriter, "write_chunk", write)
    monkeypatch.setattr(Hdf5TraceWriter, "checkpoint", checkpoint)
    original_update = jobs.update

    def update(value):
        original_update(value)
        events.append(("sqlite", value.processed_traces))

    monkeypatch.setattr(jobs, "update", update)
    token = CooperativeCancelToken()

    def callback(pct):
        events.append(("callback", pct))
        if pct > 25:
            token.request_cancel()

    result = service.resume_filter_job(job.id, callback, token)
    assert result.status is JobStatus.CANCELLED
    assert events == [
        ("sqlite", 2),
        ("callback", 25),
        ("read", 4),
        ("filter", 2),
        ("write", 4),
        ("checkpoint", 4),
        ("sqlite", 4),
        ("callback", 50),
        ("sqlite", 4),
    ]
    engine.dispose()


def test_resume_retains_point_of_no_return(tmp_path, monkeypatch):
    from giecar_seismic.application.filter_jobs import CancellationWindowClosedError
    from giecar_seismic.infrastructure.storage.hdf5_writer import Hdf5TraceWriter

    engine, _, job, _ = setup_cancelled(tmp_path, chunk_size=8)
    service, _ = resumed_service(engine, [])
    original_finalize = Hdf5TraceWriter.finalize
    attempts = []

    def finalize(writer):
        with pytest.raises(CancellationWindowClosedError):
            service.cancel_job(job.id)
        attempts.append(True)
        original_finalize(writer)

    monkeypatch.setattr(Hdf5TraceWriter, "finalize", finalize)
    result = service.resume_filter_job(job.id, lambda _: None, CooperativeCancelToken())
    assert result.status is JobStatus.COMPLETED
    assert attempts == [True]
    engine.dispose()
