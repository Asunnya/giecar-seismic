import h5py
import numpy as np
import pytest

from giecar_seismic.infrastructure.storage.hdf5_writer import (
    Hdf5TraceWriter,
    Hdf5WriterError,
)


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
