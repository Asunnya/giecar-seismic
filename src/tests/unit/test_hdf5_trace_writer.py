from pathlib import Path

import h5py
import numpy as np
import pytest

from giecar_seismic.infrastructure.storage.hdf5_writer import (
    Hdf5TraceWriter,
    Hdf5WriterError,
)


def _make_writer(
    tmp_path: Path,
    expected_trace_count: int,
    n_samples: int = 4,
    chunk_size: int = 4,
    dtype: type = np.float32,
    filename: str = "output.h5",
) -> tuple[Hdf5TraceWriter, Path]:
    path = tmp_path / filename
    writer = Hdf5TraceWriter(
        output_path=path,
        expected_trace_count=expected_trace_count,
        n_samples=n_samples,
        chunk_size=chunk_size,
        dtype=dtype,
    )
    return writer, path


def test_creates_file_with_empty_traces_dataset(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    writer.close()

    with h5py.File(path, "r") as f:
        assert f["traces"].shape == (0, 4)
        assert f.attrs["expected_trace_count"] == 10
        assert f.attrs["n_samples"] == 4
        assert f.attrs["written_trace_count"] == 0
        assert bool(f.attrs["complete"]) is False


def test_write_single_chunk_persists_only_that_many_traces(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    chunk = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)

    writer.write_chunk(0, chunk)
    writer.close()

    with h5py.File(path, "r") as f:
        assert f["traces"].shape == (3, 4)
        np.testing.assert_array_equal(f["traces"][:], chunk)
        assert f.attrs["written_trace_count"] == 3


def test_write_multiple_chunks_incrementally_persists_all_content(tmp_path):
    writer, path = _make_writer(
        tmp_path, expected_trace_count=10, n_samples=4, chunk_size=4
    )
    full = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)

    writer.write_chunk(0, full[0:4])
    writer.write_chunk(4, full[4:8])
    writer.write_chunk(8, full[8:10])
    writer.close()

    with h5py.File(path, "r") as f:
        assert f["traces"].shape == (10, 4)
        np.testing.assert_array_equal(f["traces"][:], full)
        assert f.attrs["written_trace_count"] == 10


def test_persisted_dtype_is_float32_even_for_float64_input(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=2, n_samples=4)
    chunk = np.ones((2, 4), dtype=np.float64) * 1.5

    writer.write_chunk(0, chunk)
    writer.close()

    with h5py.File(path, "r") as f:
        assert f["traces"].dtype == np.float32
        np.testing.assert_allclose(f["traces"][:], chunk.astype(np.float32))


def test_write_chunk_rejects_wrong_sample_count(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    wrong_shape_chunk = np.zeros((2, 5), dtype=np.float32)

    with pytest.raises(Hdf5WriterError):
        writer.write_chunk(0, wrong_shape_chunk)

    writer.close()


def test_write_chunk_rejects_non_2d_array(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    one_d_chunk = np.zeros(4, dtype=np.float32)

    with pytest.raises(Hdf5WriterError):
        writer.write_chunk(0, one_d_chunk)

    writer.close()


def test_write_chunk_rejects_a_gap_before_the_next_expected_start(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(Hdf5WriterError):
        writer.write_chunk(5, np.zeros((2, 4), dtype=np.float32))

    writer.close()


def test_write_chunk_rejects_overlap_with_already_written_traces(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(Hdf5WriterError):
        writer.write_chunk(2, np.zeros((2, 4), dtype=np.float32))

    writer.close()


def test_write_chunk_rejects_writing_beyond_expected_trace_count(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    writer.write_chunk(0, np.zeros((8, 4), dtype=np.float32))

    with pytest.raises(Hdf5WriterError):
        writer.write_chunk(8, np.zeros((5, 4), dtype=np.float32))

    writer.close()


def test_finalize_marks_file_complete_when_all_traces_written(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))

    writer.finalize()
    writer.close()

    with h5py.File(path, "r") as f:
        assert bool(f.attrs["complete"]) is True
        assert f.attrs["written_trace_count"] == 4


def test_finalize_fails_and_leaves_file_partial_when_traces_are_missing(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(Hdf5WriterError):
        writer.finalize()

    writer.close()

    with h5py.File(path, "r") as f:
        assert bool(f.attrs["complete"]) is False
        assert f.attrs["written_trace_count"] == 4
        assert f["traces"].shape == (4, 4)


def test_close_without_finalize_leaves_a_valid_reopenable_partial_file(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    chunk = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    writer.write_chunk(0, chunk)

    writer.close()

    with h5py.File(path, "r") as f:
        assert bool(f.attrs["complete"]) is False
        assert f["traces"].shape == (4, 4)
        np.testing.assert_array_equal(f["traces"][:], chunk)


def test_cannot_write_after_finalize(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))
    writer.finalize()

    with pytest.raises(Hdf5WriterError):
        writer.write_chunk(4, np.zeros((1, 4), dtype=np.float32))

    writer.close()


def test_cannot_write_after_close(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))
    writer.close()

    with pytest.raises(Hdf5WriterError):
        writer.write_chunk(4, np.zeros((1, 4), dtype=np.float32))


def test_cannot_finalize_after_close(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))
    writer.close()

    with pytest.raises(Hdf5WriterError):
        writer.finalize()


def test_close_is_safe_to_call_more_than_once(tmp_path):
    writer, _path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))

    writer.close()
    writer.close()  # must not raise


def test_finalize_flushes_before_marking_the_file_complete(tmp_path):
    # Strengthened: the previous version of this test only counted
    # flush() calls, which doesn't prove *order*. This observes the value
    # of the `complete` attribute at the exact moment flush() runs.
    writer, _path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))

    complete_flag_during_flush: list[bool] = []
    original_flush = writer._file.flush

    def spy_flush() -> None:
        complete_flag_during_flush.append(bool(writer._file.attrs["complete"]))
        original_flush()

    writer._file.flush = spy_flush  # type: ignore[method-assign]

    writer.finalize()

    # `complete` must still be False at the moment flush() runs: the
    # written trace data is durably flushed *before* finalize() ever
    # claims the output is complete, not after.
    # data is flushed while complete is still False, then the marker is
    # set and flushed too -- a finalize() that returns has flushed both.
    assert complete_flag_during_flush == [False, True]
    assert bool(writer._file.attrs["complete"]) is True
    writer.close()


def test_finalize_does_not_mark_complete_when_flush_fails(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))
    original_flush = writer._file.flush

    def failing_flush() -> None:
        raise OSError("simulated flush failure")

    writer._file.flush = failing_flush  # type: ignore[method-assign]

    with pytest.raises(OSError):
        writer.finalize()

    # a finalize() that raised must never leave the file marked complete,
    # even though the previous (buggy) implementation set complete=True
    # before attempting the flush that then failed.
    assert bool(writer._file.attrs["complete"]) is False

    # restore a working flush so close() -- and the reopen below -- can
    # still succeed cleanly; close() afterwards must not promote this
    # failed finalize() to complete either.
    writer._file.flush = original_flush  # type: ignore[method-assign]
    writer.close()

    with h5py.File(path, "r") as f:
        assert bool(f.attrs["complete"]) is False


def test_written_trace_count_metadata_stays_coherent_after_reopening(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=10, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))
    writer.write_chunk(4, np.zeros((3, 4), dtype=np.float32))
    writer.close()

    with h5py.File(path, "r") as f:
        assert f.attrs["written_trace_count"] == 7
        assert f["traces"].shape[0] == 7
        assert f.attrs["written_trace_count"] == f["traces"].shape[0]


def test_large_expected_trace_count_does_not_preallocate_the_dataset(tmp_path):
    # A 25GB-scale survey must not cause the writer to materialize
    # anything close to its full volume up front -- the dataset starts at
    # zero rows regardless of how large expected_trace_count is, and only
    # grows by exactly what write_chunk() has actually written.
    writer, _path = _make_writer(
        tmp_path, expected_trace_count=50_000_000, n_samples=1500, chunk_size=256
    )

    assert writer._file["traces"].shape == (0, 1500)
    assert writer._file["traces"].maxshape == (50_000_000, 1500)

    writer.close()


@pytest.mark.parametrize(
    "invalid_expected_trace_count", [0, -1], ids=["zero", "negative"]
)
def test_constructor_rejects_non_positive_expected_trace_count(
    tmp_path, invalid_expected_trace_count
):
    path = tmp_path / "output.h5"

    with pytest.raises(Hdf5WriterError):
        Hdf5TraceWriter(
            output_path=path,
            expected_trace_count=invalid_expected_trace_count,
            n_samples=4,
        )

    assert not path.exists()


@pytest.mark.parametrize("invalid_n_samples", [0, -1], ids=["zero", "negative"])
def test_constructor_rejects_non_positive_n_samples(tmp_path, invalid_n_samples):
    path = tmp_path / "output.h5"

    with pytest.raises(Hdf5WriterError):
        Hdf5TraceWriter(
            output_path=path,
            expected_trace_count=10,
            n_samples=invalid_n_samples,
        )

    assert not path.exists()


@pytest.mark.parametrize("invalid_chunk_size", [0, -1], ids=["zero", "negative"])
def test_constructor_rejects_non_positive_chunk_size(tmp_path, invalid_chunk_size):
    path = tmp_path / "output.h5"

    with pytest.raises(Hdf5WriterError):
        Hdf5TraceWriter(
            output_path=path,
            expected_trace_count=10,
            n_samples=4,
            chunk_size=invalid_chunk_size,
        )

    assert not path.exists()


def test_finalize_rolls_back_complete_when_flushing_the_marker_fails(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)
    writer.write_chunk(0, np.zeros((4, 4), dtype=np.float32))
    original_flush = writer._file.flush
    calls: list[int] = []

    def second_flush_fails() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise OSError("simulated flush failure on the completeness marker")
        original_flush()

    writer._file.flush = second_flush_fails  # type: ignore[method-assign]

    with pytest.raises(OSError):
        writer.finalize()

    # the data flush succeeded but the marker flush did not: finalize()
    # raised, so the in-memory marker must not claim completeness either.
    assert bool(writer._file.attrs["complete"]) is False

    writer._file.flush = original_flush  # type: ignore[method-assign]
    writer.close()
    with h5py.File(path, "r") as f:
        assert bool(f.attrs["complete"]) is False


def test_writer_exposes_its_output_path_read_only(tmp_path):
    writer, path = _make_writer(tmp_path, expected_trace_count=4, n_samples=4)

    assert writer.output_path == str(path)
    with pytest.raises(AttributeError):
        writer.output_path = "/elsewhere.h5"  # type: ignore[misc]

    writer.close()
