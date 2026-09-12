import h5py
import numpy as np
import pytest

from giecar_seismic.domain.job import InvalidTransitionError, Job, JobStatus
from giecar_seismic.infrastructure.storage.hdf5_writer import (
    Hdf5TraceWriter,
    Hdf5WriterError,
)


def test_resume_preserves_logical_timestamps_and_records_new_finish():
    job = Job(1, 30, 4)
    job.start()
    created, started = job.created_at, job.started_at
    for count in (1, 2):
        job.cancel()
        assert job.finished_at is not None
        job.resume()
        assert job.status is JobStatus.RUNNING
        assert job.resume_count == count
        assert (job.created_at, job.started_at) == (created, started)
        assert job.finished_at is None
    job.complete()
    assert job.finished_at is not None


@pytest.mark.parametrize(
    "status",
    [JobStatus.CREATED, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED],
)
def test_resume_rejects_every_other_status(status):
    job = Job(1, 30, 4, status=status)
    with pytest.raises(InvalidTransitionError):
        job.resume()
    assert job.resume_count == 0


@pytest.mark.parametrize("count", [-1, 11, 2, 3.5, True])
def test_checkpoint_rejects_invalid_or_regressing_counts(count):
    job = Job(1, 30, 4)
    job.start()
    job.record_checkpoint(3, 10)
    with pytest.raises(ValueError):
        job.record_checkpoint(count, 10)
    assert job.processed_traces == 3
    assert job.progress == 30


def test_checkpoint_reopen_and_resume_preserve_prefix(tmp_path):
    path = tmp_path / "partial.h5"
    prefix = np.arange(128, dtype=np.float32).reshape(2, 64)
    with Hdf5TraceWriter(path, 5, 64) as writer:
        writer.write_chunk(0, prefix)
        writer.checkpoint()
        with h5py.File(path, "r") as reopened:
            assert reopened.attrs["written_trace_count"] == 2
            np.testing.assert_array_equal(reopened["traces"][:], prefix)
    with Hdf5TraceWriter.open_for_resume(path, 5, 64) as writer:
        assert writer.written_trace_count == 2
        for start in (0, 1, 3):
            with pytest.raises(Hdf5WriterError):
                writer.write_chunk(start, np.zeros((1, 64)))
        writer.write_chunk(2, np.zeros((3, 64)))
        writer.checkpoint()
        writer.finalize()
    with h5py.File(path, "r") as reopened:
        np.testing.assert_array_equal(reopened["traces"][:2], prefix)
        assert reopened.attrs["complete"]
    with pytest.raises(Hdf5WriterError, match="complete"):
        Hdf5TraceWriter.open_for_resume(path, 5, 64)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("expected_trace_count", 6),
        ("n_samples", 65),
        ("written_trace_count", -1),
        ("written_trace_count", 6),
        ("written_trace_count", 1),
        ("written_trace_count", 2.5),
    ],
)
def test_resume_rejects_inconsistent_hdf5_metadata(tmp_path, attribute, value):
    path = tmp_path / "partial.h5"
    with Hdf5TraceWriter(path, 5, 64) as writer:
        writer.write_chunk(0, np.zeros((2, 64)))
        writer.checkpoint()
    with h5py.File(path, "r+") as file:
        file.attrs[attribute] = value
    with pytest.raises(Hdf5WriterError):
        Hdf5TraceWriter.open_for_resume(path, 5, 64)


@pytest.mark.parametrize(
    "corruption", ["missing_traces", "missing_marker", "shape", "complete"]
)
def test_resume_rejects_missing_or_invalid_structure(tmp_path, corruption):
    path = tmp_path / "partial.h5"
    with Hdf5TraceWriter(path, 5, 64) as writer:
        writer.checkpoint()
    with h5py.File(path, "r+") as file:
        if corruption == "missing_traces":
            del file["traces"]
        elif corruption == "missing_marker":
            del file.attrs["written_trace_count"]
        elif corruption == "shape":
            del file["traces"]
            file.create_dataset(
                "traces", shape=(0, 63), maxshape=(5, 63), dtype=np.float32
            )
        else:
            file.attrs["complete"] = "false"
    with pytest.raises(Hdf5WriterError):
        Hdf5TraceWriter.open_for_resume(path, 5, 64)


def test_complete_reconciliation_writer_is_immutable(tmp_path):
    path = tmp_path / "complete.h5"
    with Hdf5TraceWriter(path, 2, 64) as writer:
        writer.write_chunk(0, np.ones((2, 64)))
        writer.checkpoint()
        writer.finalize()
    with Hdf5TraceWriter.open_for_resume(path, 2, 64, allow_complete=True) as writer:
        assert writer.is_complete
        for operation in (
            lambda: writer.write_chunk(2, np.zeros((1, 64))),
            writer.checkpoint,
            writer.finalize,
        ):
            with pytest.raises(Hdf5WriterError):
                operation()
    with h5py.File(path, "r") as file:
        np.testing.assert_array_equal(file["traces"][:], np.ones((2, 64)))
