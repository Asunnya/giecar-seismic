"""Business-layer end-to-end: real SEG-Y -> streaming chunks -> Butterworth
-> incremental HDF5, with Dataset/Job persisted in a real SQLite file.

Generic by construction: every expectation below is derived from the
fixture file itself (segyio) and from the domain/filter code, never
hardcoded to one survey's numbers -- so the same assertions hold for any
SEG-Y this pipeline could process. The fixture builder takes arbitrary
inline/crossline lists, and the tests are parametrized over different
(irregular) footprints, sample counts and chunk sizes.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import segyio

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter
from giecar_seismic.application.dataset_import import ImportDatasetUseCase
from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.domain.job import JobStatus
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
)
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset
from giecar_seismic.infrastructure.segy.reader import open_dataset_reader
from giecar_seismic.infrastructure.storage.hdf5_writer import (
    job_output_path,
    make_hdf5_writer_factory,
)

# --- fixture: a small but real SEG-Y with an arbitrary footprint ----------


@dataclass(frozen=True)
class Footprint:
    """Any (inline, crossline) pairs -- irregular grids included."""

    inlines: tuple[int, ...]
    crosslines: tuple[int, ...]

    @property
    def trace_count(self) -> int:
        return len(self.inlines)


FOOTPRINTS = [
    # 3x3 grid with one position missing: 8 traces, not 9
    Footprint((10, 10, 10, 11, 11, 12, 12, 12), (1, 2, 3, 1, 2, 1, 2, 3)),
    # 2x4 envelope, only 5 traces, out-of-order acquisition
    Footprint((5, 6, 5, 6, 5), (40, 40, 41, 43, 42)),
    # a single inline: 7 traces
    Footprint((1,) * 7, tuple(range(100, 107))),
]


def write_segy(
    path: Path, footprint: Footprint, n_samples: int, sample_interval_us: int
) -> np.ndarray:
    """Creates a SEG-Y with deterministic, non-trivial amplitudes and
    returns them (traces x samples) so a test can compute its own expected
    output. Fine for a tiny fixture -- bounded memory is a constraint on
    the production pipeline, not on this test's reference data.
    """
    rng = np.random.default_rng(seed=footprint.trace_count * n_samples)
    amplitudes = rng.standard_normal((footprint.trace_count, n_samples)).astype(
        np.float32
    )
    spec = segyio.spec()
    spec.samples = list(range(n_samples))
    spec.tracecount = footprint.trace_count
    spec.format = 5  # IEEE float32
    with segyio.create(str(path), spec) as segy:
        for i, (inline, crossline) in enumerate(
            zip(footprint.inlines, footprint.crosslines)
        ):
            segy.trace[i] = amplitudes[i]
            segy.header[i][segyio.TraceField.INLINE_3D] = inline
            segy.header[i][segyio.TraceField.CROSSLINE_3D] = crossline
        segy.bin[segyio.BinField.Interval] = sample_interval_us
    return amplitudes


def segy_facts(path: Path) -> tuple[int, int, float]:
    """(physical trace count, samples per trace, sample rate ms) straight
    from the file -- the ground truth every expectation is derived from.
    """
    with segyio.open(str(path), mode="r", ignore_geometry=True) as segy:
        return segy.tracecount, len(segy.samples), float(segyio.tools.dt(segy)) / 1000


# --- composition (mirrors __main__.py, with temporary paths) --------------


def compose(db_path: Path, outputs_dir: Path, chunk_size: int):
    engine = create_sqlite_engine(db_path)
    create_schema(engine)
    factory = make_session_factory(engine)
    datasets = SqlAlchemyDatasetRepository(factory)
    jobs = SqlAlchemyJobRepository(factory)
    import_dataset = ImportDatasetUseCase(import_segy_dataset, datasets)
    service = FilterJobService(
        datasets=datasets,
        jobs=jobs,
        reader_factory=open_dataset_reader,
        writer_factory=make_hdf5_writer_factory(outputs_dir),
        chunk_size=chunk_size,
    )
    return engine, import_dataset, service


# --- tests ------------------------------------------------------------------


@pytest.mark.parametrize("footprint", FOOTPRINTS, ids=lambda f: f"{f.trace_count}tr")
@pytest.mark.parametrize("n_samples", [64, 101])
@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
def test_segy_to_hdf5_pipeline_end_to_end(
    tmp_path: Path, footprint: Footprint, n_samples: int, chunk_size: int
):
    segy_path = tmp_path / "survey.segy"
    amplitudes = write_segy(segy_path, footprint, n_samples, sample_interval_us=4000)
    n_traces, samples_per_trace, sample_rate_ms = segy_facts(segy_path)
    db_path = tmp_path / "giecar.sqlite"
    outputs_dir = tmp_path / "outputs"
    cutoff_hz, order = 30.0, 4

    engine, import_dataset, service = compose(db_path, outputs_dir, chunk_size)

    dataset = import_dataset(str(segy_path), "survey")
    assert dataset.id is not None
    assert dataset.n_traces == n_traces  # physical, from the file
    assert dataset.n_traces != 0
    assert dataset.n_samples == samples_per_trace
    assert dataset.sample_rate_ms == sample_rate_ms

    job = service.create_filter_job(dataset.id, cutoff_hz=cutoff_hz, order=order)
    assert job.id is not None

    progress_seen: list[float] = []
    finished = service.run_filter_job(
        job.id,
        progress_callback=progress_seen.append,
        cancel_token=CooperativeCancelToken(),
    )

    assert finished.status is JobStatus.COMPLETED
    assert finished.progress == 100
    assert finished.started_at is not None
    assert finished.finished_at is not None
    assert finished.created_at <= finished.started_at <= finished.finished_at
    assert finished.output_path == str(job_output_path(outputs_dir, job.id))
    assert progress_seen == sorted(progress_seen)
    assert progress_seen[-1] == 100
    # one progress emission per chunk, from the physical trace count
    assert len(progress_seen) == -(-n_traces // chunk_size)

    engine.dispose()

    # --- everything closed: reopen SQLite and HDF5 from disk ---
    fresh_jobs = SqlAlchemyJobRepository(
        make_session_factory(create_sqlite_engine(db_path))
    )
    reloaded = fresh_jobs.get(job.id)
    assert reloaded is not None
    assert reloaded.status is JobStatus.COMPLETED
    assert reloaded.progress == 100
    assert reloaded.output_path == finished.output_path
    assert reloaded.started_at == finished.started_at
    assert reloaded.finished_at == finished.finished_at

    assert reloaded.output_path is not None
    output = Path(reloaded.output_path)
    assert output.exists()
    with h5py.File(output, "r") as f:
        assert bool(f.attrs["complete"]) is True
        assert f.attrs["written_trace_count"] == n_traces
        assert f.attrs["expected_trace_count"] == n_traces
        traces = f["traces"]
        assert traces.shape == (n_traces, samples_per_trace)
        assert traces.dtype == np.float32
        expected = apply_lowpass_filter(amplitudes, cutoff_hz, order, sample_rate_ms)
        np.testing.assert_allclose(
            traces[:], expected.astype(np.float32), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("footprint", FOOTPRINTS, ids=lambda f: f"{f.trace_count}tr")
def test_cancelled_job_leaves_a_partial_unfinalized_hdf5(
    tmp_path: Path, footprint: Footprint
):
    segy_path = tmp_path / "survey.segy"
    write_segy(segy_path, footprint, n_samples=64, sample_interval_us=4000)
    n_traces, samples_per_trace, _ = segy_facts(segy_path)
    outputs_dir = tmp_path / "outputs"
    chunk_size = 2
    engine, import_dataset, service = compose(
        tmp_path / "giecar.sqlite", outputs_dir, chunk_size
    )

    dataset = import_dataset(str(segy_path), "survey")
    assert dataset.id is not None
    job = service.create_filter_job(dataset.id, cutoff_hz=30.0, order=4)
    assert job.id is not None

    # cancel right after the first chunk is reported -- from the
    # progress callback, exactly where a GUI's Cancel would land.
    token = CooperativeCancelToken()

    def cancel_after_first_chunk(_pct: float) -> None:
        if not token.is_cancelled():
            service.cancel_job(job.id)

    finished = service.run_filter_job(
        job.id, progress_callback=cancel_after_first_chunk, cancel_token=token
    )
    engine.dispose()

    assert finished.status is JobStatus.CANCELLED
    assert finished.finished_at is not None
    written_so_far = min(chunk_size, n_traces)
    if written_so_far == n_traces:
        pytest.skip("fixture fits in one chunk: nothing left to cancel before")
    assert finished.progress == pytest.approx(100 * written_so_far / n_traces)
    assert finished.progress < 100
    assert finished.output_path is not None

    with h5py.File(finished.output_path, "r") as f:
        assert bool(f.attrs["complete"]) is False
        assert f.attrs["written_trace_count"] == written_so_far
        assert f["traces"].shape == (written_so_far, samples_per_trace)
