"""SeismicViewerService: sections, display modes, trace selection and
spectrum, against fakes. No Qt, no matplotlib, no files.

Fixture geometry (3x3 envelope, (11, 3) missing -> 8 physical traces):

        xl=1  xl=2  xl=3
 il=10   t0    t1    t2
 il=11   t3    t4    --
 il=12   t5    t6    t7
"""

import numpy as np
import pytest

from giecar_seismic.application.butterworth_filter import apply_butterworth_filter
from giecar_seismic.application.filter_jobs import InvalidFilterParametersError
from giecar_seismic.application.seismic_viewer import (
    FilterPreview,
    SectionTooLargeError,
    SeismicSection,
    SeismicViewerService,
    ViewerJobNotViewableError,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation, TraceGeometry
from giecar_seismic.domain.job import FilterType, Job, JobStatus

HEADERS = [(10, 1), (10, 2), (10, 3), (11, 1), (11, 2), (12, 1), (12, 2), (12, 3)]
N_SAMPLES = 32
SAMPLE_RATE_MS = 4.0
GEOMETRY = [TraceGeometry(i, il, xl) for i, (il, xl) in enumerate(HEADERS)]


class FakeGeometryRepository:
    def traces_for_inline(self, dataset_id, inline):
        return sorted(
            (g for g in GEOMETRY if g.inline == inline), key=lambda g: g.crossline
        )

    def traces_for_crossline(self, dataset_id, crossline):
        return sorted(
            (g for g in GEOMETRY if g.crossline == crossline), key=lambda g: g.inline
        )

    def inline_numbers(self, dataset_id):
        return sorted({g.inline for g in GEOMETRY})

    def crossline_numbers(self, dataset_id):
        return sorted({g.crossline for g in GEOMETRY})

    def get_trace(self, dataset_id, trace_index):
        return next((g for g in GEOMETRY if g.trace_index == trace_index), None)


class FakeSelectiveReader:
    """Serves traces from an in-memory array and records what was asked,
    so tests can prove only the section's indices were read.
    """

    def __init__(self, data: np.ndarray):
        self._data = data
        self.requests: list[list[int]] = []
        self.closed = False

    def read_traces(self, trace_indices):
        self.requests.append(list(trace_indices))
        return self._data[list(trace_indices)]

    def close(self):
        self.closed = True


class OneItemRepository:
    def __init__(self, item):
        self._item = item

    def get(self, _id):
        return self._item if self._item.id == _id else None


def _dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="s",
        source_path="/s.segy",
        n_inlines=3,
        n_crosslines=3,
        n_traces=8,
        n_samples=N_SAMPLES,
        sample_rate_ms=SAMPLE_RATE_MS,
    )


def _job(status=JobStatus.COMPLETED, output_path="/out/job-7.h5") -> Job:
    return Job(
        id=7,
        dataset_id=1,
        cutoff_hz=30.0,
        order=4,
        status=status,
        output_path=output_path,
        progress=100.0,
    )


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    original = rng.standard_normal((8, N_SAMPLES)).astype(np.float32)
    filtered = (original * 0.5).astype(np.float32)
    readers = {"original": [], "filtered": []}

    def original_factory(dataset):
        r = FakeSelectiveReader(original)
        readers["original"].append(r)
        return r

    def filtered_factory(job):
        r = FakeSelectiveReader(filtered)
        readers["filtered"].append(r)
        return r

    service = SeismicViewerService(
        datasets=OneItemRepository(_dataset()),
        jobs=OneItemRepository(_job()),
        geometry=FakeGeometryRepository(),
        original_reader_factory=original_factory,
        filtered_reader_factory=filtered_factory,
    )
    return service, original, filtered, readers


# --- sections -------------------------------------------------------------


def test_inline_section_axis_is_crossline_with_missing_position_as_nan(world):
    service, original, _, _ = world

    section = service.load_section(7, LineOrientation.INLINE, 11)

    assert section.orientation is LineOrientation.INLINE
    assert section.line_number == 11
    assert list(section.coordinates) == [1, 2, 3]  # full survey crossline axis
    assert list(section.physical_trace_indices) == [3, 4, -1]
    assert section.original.shape == (3, N_SAMPLES)
    assert section.filtered.shape == (3, N_SAMPLES)
    np.testing.assert_array_equal(section.original[0], original[3])
    np.testing.assert_array_equal(section.original[1], original[4])
    assert np.isnan(section.original[2]).all()  # gap, neighbours not shifted
    assert np.isnan(section.filtered[2]).all()
    assert list(section.present_mask) == [True, True, False]
    assert section.sample_rate_ms == SAMPLE_RATE_MS


def test_crossline_section_axis_is_inline_ordered(world):
    service, original, _, _ = world

    section = service.load_section(7, LineOrientation.CROSSLINE, 3)

    assert list(section.coordinates) == [10, 11, 12]
    assert list(section.physical_trace_indices) == [2, -1, 7]
    np.testing.assert_array_equal(section.original[2], original[7])
    assert np.isnan(section.original[1]).all()


def test_only_the_lines_traces_are_read_and_readers_are_closed(world):
    service, _, _, readers = world

    service.load_section(7, LineOrientation.INLINE, 12)

    assert readers["original"][-1].requests == [[5, 6, 7]]
    assert readers["filtered"][-1].requests == [[5, 6, 7]]
    assert readers["original"][-1].closed and readers["filtered"][-1].closed


def test_display_modes_are_derived_from_the_loaded_section_only(world):
    service, original, filtered, _ = world
    section = service.load_section(7, LineOrientation.INLINE, 10)

    np.testing.assert_array_equal(section.original, original[[0, 1, 2]])
    np.testing.assert_array_equal(section.filtered, filtered[[0, 1, 2]])
    np.testing.assert_allclose(
        section.difference, filtered[[0, 1, 2]] - original[[0, 1, 2]]
    )


def test_time_axis_is_sample_index_times_sample_rate_ms(world):
    service, *_ = world
    section = service.load_section(7, LineOrientation.INLINE, 10)

    assert section.time_ms.shape == (N_SAMPLES,)
    np.testing.assert_allclose(section.time_ms, np.arange(N_SAMPLES) * SAMPLE_RATE_MS)


def test_line_numbers_come_from_the_geometry_index(world):
    service, *_ = world
    assert service.line_numbers(1, LineOrientation.INLINE) == [10, 11, 12]
    assert service.line_numbers(1, LineOrientation.CROSSLINE) == [1, 2, 3]


def test_section_size_guardrail_rejects_before_reading_anything(world):
    service, _, _, readers = world
    service.max_section_traces = 2

    with pytest.raises(SectionTooLargeError):
        service.load_section(7, LineOrientation.INLINE, 10)

    assert readers["original"] == [] and readers["filtered"] == []


@pytest.mark.parametrize(
    "job",
    [
        _job(status=JobStatus.CANCELLED),
        _job(status=JobStatus.RUNNING),
        _job(output_path=None),
    ],
    ids=["cancelled", "running", "no-output"],
)
def test_only_completed_jobs_with_an_output_are_viewable(world, job):
    service, *_ = world
    service._jobs = OneItemRepository(job)

    with pytest.raises(ViewerJobNotViewableError):
        service.load_section(7, LineOrientation.INLINE, 10)


# --- trace selection ---------------------------------------------------------


def test_selecting_a_coordinate_resolves_the_nearest_existing_trace(world):
    service, original, filtered, _ = world
    section = service.load_section(7, LineOrientation.INLINE, 12)

    view = service.select_trace(7, section, coordinate=2.3)  # nearest xl=2

    assert view is not None
    assert view.geometry == TraceGeometry(6, 12, 2)
    assert view.dataset.id == 1 and view.job.id == 7
    np.testing.assert_array_equal(view.original, original[6])
    np.testing.assert_array_equal(view.filtered, filtered[6])
    np.testing.assert_allclose(view.time_ms, np.arange(N_SAMPLES) * SAMPLE_RATE_MS)


def test_selecting_a_missing_position_does_not_invent_a_trace(world):
    service, *_ = world
    section = service.load_section(7, LineOrientation.INLINE, 11)

    assert service.select_trace(7, section, coordinate=3.0) is None  # (11,3) absent
    assert service.select_trace(7, section, coordinate=2.0) is not None


def test_selecting_outside_the_axis_returns_none(world):
    service, *_ = world
    section = service.load_section(7, LineOrientation.INLINE, 10)

    assert service.select_trace(7, section, coordinate=42.0) is None


# --- spectrum ---------------------------------------------------------------


def test_spectrum_uses_fs_from_sample_rate_and_stops_at_nyquist(world):
    service, *_ = world
    section = service.load_section(7, LineOrientation.INLINE, 10)
    view = service.select_trace(7, section, coordinate=1.0)
    assert view is not None

    spectrum = service.spectrum(view)

    fs = 1000 / SAMPLE_RATE_MS
    assert spectrum.nyquist_hz == fs / 2 == 125.0
    assert spectrum.frequencies_hz[0] == 0
    assert spectrum.frequencies_hz[-1] == pytest.approx(spectrum.nyquist_hz)
    assert (
        spectrum.frequencies_hz.shape
        == spectrum.original.shape
        == spectrum.filtered.shape
    )
    np.testing.assert_allclose(
        spectrum.original, np.abs(np.fft.rfft(view.original)), rtol=1e-5
    )
    np.testing.assert_allclose(
        spectrum.filtered, np.abs(np.fft.rfft(view.filtered)), rtol=1e-5
    )


def test_spectrum_carries_the_jobs_cutoff_for_presentation(world):
    service, *_ = world
    section = service.load_section(7, LineOrientation.INLINE, 10)
    view = service.select_trace(7, section, coordinate=1.0)
    assert view is not None

    assert service.spectrum(view).cutoff_hz == 30.0


def test_section_is_a_plain_value_object_without_qt_or_matplotlib():
    import inspect

    import giecar_seismic.application.seismic_viewer as module

    import_lines = [
        line
        for line in inspect.getsource(module).splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert not any("PyQt5" in line or "matplotlib" in line for line in import_lines)
    assert SeismicSection.__dataclass_fields__.keys() >= {
        "orientation",
        "line_number",
        "coordinates",
        "physical_trace_indices",
        "original",
        "filtered",
        "sample_rate_ms",
    }


# --- preview (filter parameters on a dataset, no job, no HDF5) ---------------


def _preview(**overrides) -> FilterPreview:
    params = {"dataset_id": 1, "cutoff_hz": 30.0, "order": 4}
    params.update(overrides)
    return FilterPreview(**params)


def test_preview_context_is_an_unpersisted_job_flagged_as_preview(world):
    service, *_ = world

    ctx = service.context(_preview(filter_type=FilterType.HIGH_PASS))

    assert ctx.preview is True
    assert ctx.dataset.id == 1
    assert ctx.job.id is None
    assert ctx.job.status is JobStatus.CREATED
    assert (ctx.job.cutoff_hz, ctx.job.order, ctx.job.filter_type) == (
        30.0,
        4,
        FilterType.HIGH_PASS,
    )


def test_job_context_is_not_a_preview(world):
    service, *_ = world
    assert service.context(7).preview is False


def test_preview_rejects_parameters_the_job_would_reject(world):
    service, *_ = world
    nyquist = 500 / SAMPLE_RATE_MS
    with pytest.raises(InvalidFilterParametersError, match="Nyquist"):
        service.context(_preview(cutoff_hz=nyquist))
    with pytest.raises(InvalidFilterParametersError, match="order"):
        service.context(_preview(order=9))
    with pytest.raises(InvalidFilterParametersError, match="upper_cutoff_hz"):
        service.context(_preview(filter_type=FilterType.BAND_PASS))


def test_preview_of_unknown_dataset_is_not_viewable(world):
    service, *_ = world
    with pytest.raises(ViewerJobNotViewableError, match="dataset 99"):
        service.context(_preview(dataset_id=99))


@pytest.mark.parametrize(
    "preview",
    [
        _preview(),
        _preview(filter_type=FilterType.HIGH_PASS),
        _preview(
            filter_type=FilterType.BAND_PASS, cutoff_hz=20.0, upper_cutoff_hz=50.0
        ),
    ],
    ids=["low", "high", "band"],
)
def test_preview_section_filters_the_line_in_memory_and_never_opens_the_output(
    world, preview
):
    service, original, _, readers = world

    section = service.load_section(preview, LineOrientation.INLINE, 12)

    assert readers["filtered"] == []  # no HDF5 / job output touched
    assert readers["original"][0].requests == [[5, 6, 7]]  # only this line's traces
    assert section.filtered.dtype == np.float32
    expected = apply_butterworth_filter(
        original[[5, 6, 7]],
        preview.cutoff_hz,
        preview.order,
        SAMPLE_RATE_MS,
        filter_type=preview.filter_type,
        upper_cutoff_hz=preview.upper_cutoff_hz,
    )
    np.testing.assert_allclose(section.filtered, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(section.original, original[[5, 6, 7]])


def test_preview_section_keeps_gaps_as_nan(world):
    service, *_ = world
    section = service.load_section(_preview(), LineOrientation.INLINE, 11)
    assert list(section.physical_trace_indices) == [3, 4, -1]
    assert np.isnan(section.filtered[2]).all()
    assert np.isfinite(section.filtered[:2]).all()


def test_preview_trace_selection_and_spectrum_carry_the_preview_parameters(world):
    service, *_ = world
    preview = _preview(
        filter_type=FilterType.BAND_PASS, cutoff_hz=20.0, upper_cutoff_hz=50.0
    )
    section = service.load_section(preview, LineOrientation.INLINE, 10)

    view = service.select_trace(preview, section, coordinate=2.0)

    assert view is not None
    assert view.job.id is None
    assert view.geometry.trace_index == 1
    spectrum = service.spectrum(view)
    assert [m.frequency_hz for m in spectrum.cutoff_markers] == [20.0, 50.0]
    assert spectrum.filter_response is not None
