from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

import numpy as np
from scipy.signal import sosfreqz

from giecar_seismic.application.butterworth_filter import (
    apply_butterworth_filter,
    butterworth_sos,
)
from giecar_seismic.application.filter_jobs import (
    DatasetRepository,
    JobRepository,
    validate_filter_parameters,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation, TraceGeometry
from giecar_seismic.domain.job import FilterType, Job, JobStatus

# Guardrail for surveys much larger than the sample: one section is
# (traces on the line) x n_samples and must stay a few MB, never the
# volume. Beyond this, a future display-decimation step would be needed.
DEFAULT_MAX_SECTION_TRACES = 5000


class GeometryRepository(Protocol):
    """Read side of the geometry index. Every call returns one line's rows
    or a distinct, sorted list of line numbers -- never the whole survey.
    """

    def traces_for_inline(
        self, dataset_id: int, inline: int
    ) -> list[TraceGeometry]: ...
    def traces_for_crossline(
        self, dataset_id: int, crossline: int
    ) -> list[TraceGeometry]: ...
    def inline_numbers(self, dataset_id: int) -> list[int]: ...
    def crossline_numbers(self, dataset_id: int) -> list[int]: ...
    def get_trace(self, dataset_id: int, trace_index: int) -> TraceGeometry | None: ...


class SelectiveTraceReader(Protocol):
    """Reads an arbitrary small set of physical traces (SegyTraceReader and
    Hdf5TraceReader both satisfy this)."""

    def read_traces(self, trace_indices: Sequence[int]) -> np.ndarray: ...
    def close(self) -> None: ...


OriginalReaderFactory = Callable[[SeismicDataset], SelectiveTraceReader]
FilteredReaderFactory = Callable[[Job], SelectiveTraceReader]


class SectionTooLargeError(Exception):
    pass


class ViewerJobNotViewableError(Exception):
    """Only a COMPLETED job with an output_path has a finalized HDF5 to
    view against its SEG-Y. (Partial CANCELLED outputs could be allowed
    later; keeping the first version's semantics simple.)"""


@dataclass(frozen=True)
class FilterPreview:
    """Filter parameters to preview on a dataset *before* any job exists.

    Nothing is persisted and no HDF5 is produced: the viewer applies the
    filter in memory to the one section on screen (bounded by
    max_section_traces x n_samples), using the same SOS design the job
    would use, so what is shown is what run_filter_job() would write for
    that line.
    """

    dataset_id: int
    cutoff_hz: float
    order: int
    filter_type: FilterType = FilterType.LOW_PASS
    upper_cutoff_hz: float | None = None


# What the viewer looks at: a persisted job's output, or a preview.
ViewerTarget = int | FilterPreview


@dataclass(frozen=True)
class SeismicSection:
    """One inline or crossline, original and filtered, on the survey's
    full coordinate axis.

    For an inline, `coordinates` are *all* crossline numbers of the survey
    (and vice versa), so a missing (inline, crossline) position stays a
    gap: its row is NaN and its physical index is -1, and neighbouring
    traces never shift. Arrays are (len(coordinates), n_samples) --
    proportional to one line, never to the volume.
    """

    orientation: LineOrientation
    line_number: int
    coordinates: np.ndarray  # int, sorted, full survey axis
    physical_trace_indices: np.ndarray  # int, -1 where missing
    original: np.ndarray  # float32, NaN where missing
    filtered: np.ndarray  # float32, NaN where missing
    sample_rate_ms: float

    @property
    def n_samples(self) -> int:
        return int(self.original.shape[1])

    @property
    def time_ms(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.sample_rate_ms

    @property
    def present_mask(self) -> np.ndarray:
        return self.physical_trace_indices >= 0

    @property
    def difference(self) -> np.ndarray:
        return self.filtered - self.original

    @property
    def n_present_traces(self) -> int:
        return int(self.present_mask.sum())


@dataclass(frozen=True)
class TraceView:
    geometry: TraceGeometry
    time_ms: np.ndarray
    original: np.ndarray
    filtered: np.ndarray
    dataset: SeismicDataset
    job: Job


class SpectrumScale(Enum):
    LINEAR = "Linear"
    DB = "dB"


@dataclass(frozen=True)
class CutoffMarker:
    frequency_hz: float
    label: str


@dataclass(frozen=True)
class TraceSpectrum:
    frequencies_hz: np.ndarray  # 0 .. nyquist
    original: np.ndarray  # |rfft|
    filtered: np.ndarray
    nyquist_hz: float
    cutoff_hz: float
    filter_type: FilterType = FilterType.LOW_PASS
    upper_cutoff_hz: float | None = None
    filter_response: np.ndarray | None = None  # zero-phase gain, |H|²
    scale: SpectrumScale = SpectrumScale.LINEAR
    show_filter_response: bool = False
    reference_magnitude: float = 1.0

    @property
    def cutoff_markers(self) -> tuple[CutoffMarker, ...]:
        if self.filter_type is FilterType.LOW_PASS:
            return (CutoffMarker(self.cutoff_hz, "High cutoff"),)
        low = CutoffMarker(self.cutoff_hz, "Low cutoff")
        if self.filter_type is FilterType.BAND_PASS:
            assert self.upper_cutoff_hz is not None
            return (low, CutoffMarker(self.upper_cutoff_hz, "High cutoff"))
        return (low,)

    @property
    def magnitude_label(self) -> str:
        return "Magnitude (dB)" if self.scale is SpectrumScale.DB else "Magnitude"

    @property
    def response_label(self) -> str:
        return "Filter gain (dB)" if self.scale is SpectrumScale.DB else "Filter gain"


def _magnitude_db(magnitude: np.ndarray, reference: float) -> np.ndarray:
    # Work in log space to avoid under/overflow when dividing tiny/large
    # magnitudes. Clamp at -120 dB (amplitude ratio 1e-6), never log(0).
    tiny = np.finfo(np.float64).tiny
    return np.maximum(
        20 * (np.log10(np.maximum(magnitude, tiny)) - np.log10(reference)), -120.0
    )


def spectrum_for_display(
    spectrum: TraceSpectrum, scale: SpectrumScale, *, show_filter_response: bool = False
) -> TraceSpectrum:
    """Transform a cached *linear* spectrum without FFT, I/O or workers.

    Both curves use the original's peak as the same dB reference (0 dB).
    For an all-zero original, use fixed reference 1. Response is a gain
    relative to unity and is drawn on a separate Y axis, never normalized
    to either trace. Always transform the raw spectrum, not a previous dB
    result, so repeated toggles cannot accumulate conversion errors.
    """
    if spectrum.scale is not SpectrumScale.LINEAR:
        raise ValueError("spectrum_for_display requires a raw linear spectrum")
    peak = float(np.max(spectrum.original))
    reference = max(peak, np.finfo(np.float64).tiny) if peak > 0 else 1.0
    original, filtered, response = (
        spectrum.original,
        spectrum.filtered,
        spectrum.filter_response,
    )
    if scale is SpectrumScale.DB:
        original = _magnitude_db(original, reference)
        filtered = _magnitude_db(filtered, reference)
        if response is not None:
            response = _magnitude_db(response, 1.0)
    return replace(
        spectrum,
        original=original,
        filtered=filtered,
        filter_response=response,
        scale=scale,
        show_filter_response=show_filter_response,
        reference_magnitude=reference,
    )


@dataclass(frozen=True)
class ViewerContext:
    dataset: SeismicDataset
    # For a preview this is an unpersisted Job (id None) carrying only the
    # filter parameters -- the single shape the spectrum/labels consume.
    job: Job
    preview: bool = False


class SeismicViewerService:
    """Qt- and matplotlib-free use cases behind the 2D viewer.

    Resolves a job's dataset, assembles one line at a time from the
    geometry index plus selective SEG-Y/HDF5 reads, and derives per-trace
    views and spectra from an already-loaded section (no further I/O).
    """

    def __init__(
        self,
        datasets: DatasetRepository,
        jobs: JobRepository,
        geometry: GeometryRepository,
        original_reader_factory: OriginalReaderFactory,
        filtered_reader_factory: FilteredReaderFactory,
        max_section_traces: int = DEFAULT_MAX_SECTION_TRACES,
    ) -> None:
        self._datasets = datasets
        self._jobs = jobs
        self._geometry = geometry
        self._original_reader_factory = original_reader_factory
        self._filtered_reader_factory = filtered_reader_factory
        self.max_section_traces = max_section_traces

    def context(self, target: ViewerTarget) -> ViewerContext:
        if isinstance(target, FilterPreview):
            return self._preview_context(target)
        job_id = target
        job = self._jobs.get(job_id)
        if job is None:
            raise ViewerJobNotViewableError(f"job {job_id} not found")
        if job.status is not JobStatus.COMPLETED or job.output_path is None:
            raise ViewerJobNotViewableError(
                f"job {job_id} is {job.status.name} -- only COMPLETED jobs "
                "with an output can be viewed"
            )
        dataset = self._datasets.get(job.dataset_id)
        if dataset is None or dataset.id is None:
            raise ViewerJobNotViewableError(f"dataset {job.dataset_id} not found")
        return ViewerContext(dataset=dataset, job=job)

    def _preview_context(self, preview: FilterPreview) -> ViewerContext:
        dataset = self._datasets.get(preview.dataset_id)
        if dataset is None or dataset.id is None:
            raise ViewerJobNotViewableError(f"dataset {preview.dataset_id} not found")
        # Same rules as create_filter_job(): raises InvalidFilterParametersError.
        validate_filter_parameters(
            dataset,
            preview.cutoff_hz,
            preview.order,
            filter_type=preview.filter_type,
            upper_cutoff_hz=preview.upper_cutoff_hz,
        )
        job = Job(
            dataset_id=preview.dataset_id,
            cutoff_hz=preview.cutoff_hz,
            order=preview.order,
            filter_type=preview.filter_type,
            upper_cutoff_hz=preview.upper_cutoff_hz,
        )
        return ViewerContext(dataset=dataset, job=job, preview=True)

    def line_numbers(self, dataset_id: int, orientation: LineOrientation) -> list[int]:
        if orientation is LineOrientation.INLINE:
            return self._geometry.inline_numbers(dataset_id)
        return self._geometry.crossline_numbers(dataset_id)

    def load_section(
        self, target: ViewerTarget, orientation: LineOrientation, line_number: int
    ) -> SeismicSection:
        ctx = self.context(target)
        dataset_id = ctx.dataset.id
        assert dataset_id is not None

        if orientation is LineOrientation.INLINE:
            present = self._geometry.traces_for_inline(dataset_id, line_number)
            axis = self._geometry.crossline_numbers(dataset_id)
            coordinate_of = {g.crossline: g for g in present}
        else:
            present = self._geometry.traces_for_crossline(dataset_id, line_number)
            axis = self._geometry.inline_numbers(dataset_id)
            coordinate_of = {g.inline: g for g in present}

        if len(axis) > self.max_section_traces:
            raise SectionTooLargeError(
                f"a {orientation.value} section would span {len(axis)} traces, "
                f"above the configured limit of {self.max_section_traces}"
            )

        coordinates = np.asarray(axis, dtype=np.int64)
        physical = np.full(len(axis), -1, dtype=np.int64)
        for position, coordinate in enumerate(axis):
            geometry = coordinate_of.get(coordinate)
            if geometry is not None:
                physical[position] = geometry.trace_index
        present_positions = np.flatnonzero(physical >= 0)
        present_indices = [int(i) for i in physical[present_positions]]

        original = np.full((len(axis), ctx.dataset.n_samples), np.nan, dtype=np.float32)
        filtered = np.full((len(axis), ctx.dataset.n_samples), np.nan, dtype=np.float32)
        if present_indices:
            original[present_positions] = self._read(
                self._original_reader_factory(ctx.dataset), present_indices
            )
            if ctx.preview:
                # One section only -- never the volume -- through the exact
                # design the job would use, so the preview is faithful.
                filtered[present_positions] = apply_butterworth_filter(
                    original[present_positions],
                    ctx.job.cutoff_hz,
                    ctx.job.order,
                    ctx.dataset.sample_rate_ms,
                    filter_type=ctx.job.filter_type,
                    upper_cutoff_hz=ctx.job.upper_cutoff_hz,
                ).astype(np.float32)
            else:
                filtered[present_positions] = self._read(
                    self._filtered_reader_factory(ctx.job), present_indices
                )

        return SeismicSection(
            orientation=orientation,
            line_number=line_number,
            coordinates=coordinates,
            physical_trace_indices=physical,
            original=original,
            filtered=filtered,
            sample_rate_ms=ctx.dataset.sample_rate_ms,
        )

    @staticmethod
    def _read(reader: SelectiveTraceReader, indices: list[int]) -> np.ndarray:
        try:
            return np.asarray(reader.read_traces(indices), dtype=np.float32)
        finally:
            reader.close()

    def select_trace(
        self, target: ViewerTarget, section: SeismicSection, coordinate: float
    ) -> TraceView | None:
        """Snap a clicked coordinate to the nearest axis position. If that
        position has no physical trace (a gap), return None explicitly --
        never silently the next neighbour."""
        if section.coordinates.size == 0:
            return None
        position = int(np.argmin(np.abs(section.coordinates - coordinate)))
        if abs(float(section.coordinates[position]) - coordinate) > 0.5:
            return None  # clicked outside the axis
        trace_index = int(section.physical_trace_indices[position])
        if trace_index < 0:
            return None
        ctx = self.context(target)
        assert ctx.dataset.id is not None
        geometry = self._geometry.get_trace(ctx.dataset.id, trace_index)
        if geometry is None:
            return None
        return TraceView(
            geometry=geometry,
            time_ms=section.time_ms,
            original=section.original[position],
            filtered=section.filtered[position],
            dataset=ctx.dataset,
            job=ctx.job,
        )

    @staticmethod
    def spectrum(view: TraceView) -> TraceSpectrum:
        fs_hz = 1000 / view.dataset.sample_rate_ms
        n = view.original.shape[0]
        frequencies = np.fft.rfftfreq(n, d=1 / fs_hz)
        sos = butterworth_sos(
            view.job.cutoff_hz,
            view.job.order,
            view.dataset.sample_rate_ms,
            filter_type=view.job.filter_type,
            upper_cutoff_hz=view.job.upper_cutoff_hz,
        )
        _, response = sosfreqz(sos, worN=frequencies, fs=fs_hz)
        return TraceSpectrum(
            frequencies_hz=frequencies,
            original=np.abs(np.fft.rfft(view.original)),
            filtered=np.abs(np.fft.rfft(view.filtered)),
            nyquist_hz=fs_hz / 2,
            cutoff_hz=view.job.cutoff_hz,
            filter_type=view.job.filter_type,
            upper_cutoff_hz=view.job.upper_cutoff_hz,
            filter_response=np.abs(response) ** 2,
        )
