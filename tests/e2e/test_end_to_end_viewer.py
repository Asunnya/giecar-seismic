"""Viewer end-to-end on the business layer: irregular synthetic SEG-Y ->
geometry index -> real filter pipeline -> HDF5 -> inline/crossline
sections -> original/filtered/difference -> trace selection -> spectrum.

Every expectation is derived from the fixture (headers, amplitudes) and
from apply_lowpass_filter(); nothing is specific to the real survey.
"""

from pathlib import Path

import numpy as np
import pytest
import segyio

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter
from giecar_seismic.application.dataset_import import ImportDatasetUseCase
from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.application.geometry_index import BuildGeometryIndexUseCase
from giecar_seismic.application.seismic_viewer import (
    FilterPreview,
    SectionRegion,
    SeismicViewerService,
)
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.domain.job import JobStatus
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyGeometryRepository,
    SqlAlchemyJobRepository,
)
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset
from giecar_seismic.infrastructure.segy.reader import (
    iter_trace_header_batches,
    open_dataset_reader,
)
from giecar_seismic.infrastructure.storage.hdf5_reader import open_job_output_reader
from giecar_seismic.infrastructure.storage.hdf5_writer import make_hdf5_writer_factory

# 3x4 envelope, three positions missing, written out of grid order.
HEADERS = [
    (20, 3),
    (21, 1),
    (20, 1),
    (22, 4),
    (21, 2),
    (20, 4),
    (22, 1),
    (21, 4),
    (22, 2),
]
N_SAMPLES = 512  # long enough that filtfilt edge effects do not dominate the spectrum
CUTOFF_HZ, ORDER, DT_US = 30.0, 4, 4000


def _write_segy(path: Path) -> np.ndarray:
    rng = np.random.default_rng(1)
    amplitudes = rng.standard_normal((len(HEADERS), N_SAMPLES)).astype(np.float32)
    spec = segyio.spec()
    spec.samples = list(range(N_SAMPLES))
    spec.tracecount = len(HEADERS)
    spec.format = 5
    with segyio.create(str(path), spec) as segy:
        for i, (il, xl) in enumerate(HEADERS):
            segy.trace[i] = amplitudes[i]
            segy.header[i][segyio.TraceField.INLINE_3D] = il
            segy.header[i][segyio.TraceField.CROSSLINE_3D] = xl
        segy.bin[segyio.BinField.Interval] = DT_US
    return amplitudes


@pytest.fixture
def world(tmp_path: Path):
    segy_path = tmp_path / "s.segy"
    amplitudes = _write_segy(segy_path)
    engine = create_sqlite_engine(tmp_path / "g.sqlite")
    create_schema(engine)
    factory = make_session_factory(engine)
    datasets = SqlAlchemyDatasetRepository(factory)
    jobs = SqlAlchemyJobRepository(factory)
    geometry = SqlAlchemyGeometryRepository(factory)

    dataset = ImportDatasetUseCase(import_segy_dataset, datasets)(str(segy_path), "s")
    assert dataset.id is not None
    filter_service = FilterJobService(
        datasets=datasets,
        jobs=jobs,
        reader_factory=open_dataset_reader,
        writer_factory=make_hdf5_writer_factory(tmp_path / "out"),
        chunk_size=4,
    )
    job = filter_service.create_filter_job(dataset.id, CUTOFF_HZ, ORDER)
    assert job.id is not None
    done = filter_service.run_filter_job(
        job.id, lambda _p: None, CooperativeCancelToken()
    )
    assert done.status is JobStatus.COMPLETED

    build_index = BuildGeometryIndexUseCase(
        iter_trace_header_batches, geometry, batch_size=4
    )
    viewer = SeismicViewerService(
        datasets, jobs, geometry, open_dataset_reader, open_job_output_reader
    )
    expected_filtered = apply_lowpass_filter(
        amplitudes, CUTOFF_HZ, ORDER, DT_US / 1000
    ).astype(np.float32)
    return viewer, build_index, dataset, job, amplitudes, expected_filtered, geometry


def _by_coord(orientation, line):
    """Ground truth from the fixture: {coordinate: physical index}."""
    if orientation is LineOrientation.INLINE:
        return {xl: i for i, (il, xl) in enumerate(HEADERS) if il == line}
    return {il: i for i, (il, xl) in enumerate(HEADERS) if xl == line}


def test_viewer_end_to_end(world):
    viewer, build_index, dataset, job, amplitudes, expected_filtered, geometry = world
    assert dataset.id is not None and job.id is not None

    # geometry index: built in bounded batches, complete, idempotent
    assert build_index(dataset) is True
    assert geometry.count(dataset.id) == dataset.n_traces == len(HEADERS)
    assert dataset.n_traces != dataset.n_inlines * dataset.n_crosslines
    assert build_index(dataset) is False

    inlines = sorted({il for il, _ in HEADERS})
    crosslines = sorted({xl for _, xl in HEADERS})
    assert viewer.line_numbers(dataset.id, LineOrientation.INLINE) == inlines
    assert viewer.line_numbers(dataset.id, LineOrientation.CROSSLINE) == crosslines

    for orientation, lines, axis in (
        (LineOrientation.INLINE, inlines, crosslines),
        (LineOrientation.CROSSLINE, crosslines, inlines),
    ):
        for line in lines:
            section = viewer.load_section(job.id, orientation, line)
            truth = _by_coord(orientation, line)

            assert list(section.coordinates) == axis
            assert section.original.shape == (len(axis), N_SAMPLES)
            assert section.original.shape[0] <= max(len(inlines), len(crosslines))
            assert (
                section.original.shape[0] < dataset.n_traces
                or len(axis) == dataset.n_traces
            )
            for position, coord in enumerate(axis):
                if coord in truth:
                    idx = truth[coord]
                    assert section.physical_trace_indices[position] == idx
                    np.testing.assert_array_equal(
                        section.original[position], amplitudes[idx]
                    )
                    np.testing.assert_allclose(
                        section.filtered[position],
                        expected_filtered[idx],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                else:
                    assert section.physical_trace_indices[position] == -1
                    assert np.isnan(section.original[position]).all()
                    assert np.isnan(section.filtered[position]).all()
            present = section.present_mask
            np.testing.assert_allclose(
                section.difference[present],
                (section.filtered - section.original)[present],
            )
            np.testing.assert_allclose(
                section.time_ms, np.arange(N_SAMPLES) * DT_US / 1000
            )

    # regional QC on an inline with a gap: bounds on the axis, gap excluded,
    # curves are the mean of per-trace |rfft| of the traces really there
    line = next(
        il
        for il in inlines
        if len(_by_coord(LineOrientation.INLINE, il)) < len(crosslines)
    )
    section = viewer.load_section(job.id, LineOrientation.INLINE, line)
    truth = _by_coord(LineOrientation.INLINE, line)
    region = SectionRegion.from_boundaries(crosslines[-1], crosslines[0])
    regional = viewer.regional_spectrum(section, region, dataset, job)
    inside = [truth[xl] for xl in crosslines if xl in truth]
    assert regional.region.trace_indices == tuple(inside)
    assert regional.n_positions == len(crosslines)
    assert regional.n_present == len(inside)
    assert regional.n_missing == len(crosslines) - len(inside)
    np.testing.assert_allclose(
        regional.spectrum.original,
        np.abs(np.fft.rfft(amplitudes[inside], axis=-1)).mean(axis=0),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        regional.spectrum.filtered,
        np.abs(np.fft.rfft(expected_filtered[inside], axis=-1)).mean(axis=0),
        rtol=1e-4,
    )

    # spectrum: fs from the file's dt, up to Nyquist, cutoff from the job,
    # and the low-pass visibly attenuates above the cutoff
    spectrum = regional.spectrum
    fs = 1000 / dataset.sample_rate_ms
    assert spectrum.nyquist_hz == fs / 2
    assert spectrum.frequencies_hz[-1] == pytest.approx(fs / 2)
    assert spectrum.cutoff_hz == CUTOFF_HZ
    above = spectrum.frequencies_hz > 2 * CUTOFF_HZ
    assert spectrum.filtered[above].sum() < 0.25 * spectrum.original[above].sum()


def test_preview_matches_the_completed_jobs_output_for_every_line(world):
    """A preview is only useful if it shows what the job would produce:
    the in-memory preview section (SEG-Y -> filter, no HDF5) must match
    the section read from the finalized HDF5 of a job with the same
    parameters, on every inline and crossline, gaps included."""
    viewer, build_index, dataset, job, _, expected_filtered, _ = world
    assert dataset.id is not None and job.id is not None
    assert build_index(dataset) is True
    preview = FilterPreview(dataset.id, CUTOFF_HZ, ORDER)

    for orientation in LineOrientation:
        for line in viewer.line_numbers(dataset.id, orientation):
            from_hdf5 = viewer.load_section(job.id, orientation, line)
            previewed = viewer.load_section(preview, orientation, line)
            np.testing.assert_array_equal(
                previewed.physical_trace_indices, from_hdf5.physical_trace_indices
            )
            np.testing.assert_array_equal(previewed.original, from_hdf5.original)
            np.testing.assert_allclose(
                previewed.filtered, from_hdf5.filtered, rtol=1e-5, atol=1e-6
            )
            # and both are the reference filter of the fixture amplitudes
            present = previewed.present_mask
            np.testing.assert_allclose(
                previewed.filtered[present],
                expected_filtered[previewed.physical_trace_indices[present]],
                rtol=1e-5,
                atol=1e-6,
            )

    # the preview never became a job
    assert [j.id for j in viewer._jobs.list(dataset_id=dataset.id)] == [job.id]
