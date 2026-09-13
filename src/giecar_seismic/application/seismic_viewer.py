from __future__ import annotations

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

    def position_for_coordinate(self, coordinate: float) -> int | None:
        """Snap a clicked coordinate to the nearest axis position, or None
        when the click is outside the axis (> 0.5 from every position).
        Pure: resolving the position to a physical trace is the caller's
        decision (it may be a gap, index -1)."""
        if self.coordinates.size == 0:
            return None
        position = int(np.argmin(np.abs(self.coordinates - coordinate)))
        if abs(float(self.coordinates[position]) - coordinate) > 0.5:
            return None
        return position


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


# Regional QC needs at least this many physically present traces; a region
# of exactly one present trace IS the single-trace analysis.
MIN_REGION_TRACES = 1
# Rows per FFT batch while aggregating a region: temporary memory is
# chunk x n_frequency_bins, whatever the region's size.
REGION_SPECTRUM_CHUNK_TRACES = 64


class RegionTooSmallError(ValueError):
    """The region holds fewer than MIN_REGION_TRACES physical traces."""


@dataclass(frozen=True)
class SectionRegion:
    """A contiguous, inclusive interval of the loaded section's axis
    (crossline numbers on an inline, inline numbers on a crossline).
    Normalized: lower <= upper, whatever the click order."""

    lower_coordinate: int
    upper_coordinate: int

    def __post_init__(self) -> None:
        if self.lower_coordinate > self.upper_coordinate:
            raise ValueError("lower_coordinate must not exceed upper_coordinate")

    @classmethod
    def from_boundaries(cls, first: int, second: int) -> SectionRegion:
        return cls(min(first, second), max(first, second))

    def resolve(self, section: SeismicSection) -> ResolvedRegion:
        """Which axis positions fall inside, and which of them carry a
        physical trace. Both bounds must be actual axis coordinates (the
        viewer snaps clicks to the axis); a bound may sit on a gap."""
        coordinates = section.coordinates
        for bound in (self.lower_coordinate, self.upper_coordinate):
            if not np.any(coordinates == bound):
                raise ValueError(f"coordinate {bound} is not on the section axis")
        inside = np.flatnonzero(
            (coordinates >= self.lower_coordinate)
            & (coordinates <= self.upper_coordinate)
        )
        physical = section.physical_trace_indices[inside]
        present = inside[physical >= 0]
        return ResolvedRegion(
            region=self,
            orientation=section.orientation,
            line_number=section.line_number,
            positions=tuple(int(p) for p in inside),
            present_positions=tuple(int(p) for p in present),
            trace_indices=tuple(int(i) for i in physical[physical >= 0]),
        )


@dataclass(frozen=True)
class ResolvedRegion:
    """A SectionRegion projected onto one loaded section: axis positions
    inside it, the subset that physically exists, and the gaps."""

    region: SectionRegion
    orientation: LineOrientation
    line_number: int
    positions: tuple[int, ...]
    present_positions: tuple[int, ...]
    trace_indices: tuple[int, ...]

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def n_present(self) -> int:
        return len(self.present_positions)

    @property
    def n_missing(self) -> int:
        return self.n_positions - self.n_present

    @property
    def is_valid(self) -> bool:
        return self.n_present >= MIN_REGION_TRACES

    @property
    def is_single_trace(self) -> bool:
        return self.n_present == 1


@dataclass(frozen=True)
class TraceWaveform:
    """Amplitude x time of ONE trace, original and filtered -- shown only
    when the region is a single trace. A region is never averaged in the
    time domain, so wider regions carry no waveform."""

    trace_index: int
    coordinate: int
    time_ms: np.ndarray
    original: np.ndarray
    filtered: np.ndarray

    @property
    def amplitude_scale(self) -> float:
        """One symmetric amplitude scale for the original/filtered overlay."""
        values = np.abs(np.concatenate([self.original, self.filtered]))
        return float(np.nanmax(values) or 1.0)


@dataclass(frozen=True)
class RegionalSpectrum:
    """Ensemble QC over a region of one loaded section.

    `spectrum.original` / `spectrum.filtered` are, per frequency bin, the
    arithmetic mean of the per-trace linear amplitude spectra of every
    physically present trace in the region (TRACE -> FFT -> |.| -> mean),
    never the spectrum of the averaged traces: time-domain averaging lets
    neighbouring traces cancel by phase and misrepresents the frequency
    content. TraceSpectrum is reused as the array container so the same
    Linear/dB rules and renderers apply.
    """

    METHOD = "Mean of per-trace amplitude spectra"

    spectrum: TraceSpectrum
    region: ResolvedRegion
    dataset: SeismicDataset
    job: Job
    waveform: TraceWaveform | None = None  # only for a single-trace region

    @property
    def n_present(self) -> int:
        return self.region.n_present

    @property
    def n_missing(self) -> int:
        return self.region.n_missing

    @property
    def n_positions(self) -> int:
        return self.region.n_positions


def regional_spectrum_for_display(
    regional: RegionalSpectrum,
    scale: SpectrumScale,
    *,
    show_filter_response: bool = False,
) -> RegionalSpectrum:
    """Same Linear/dB transform as any spectrum, applied to the already
    aggregated linear magnitudes (dB after aggregation, never a mean of
    dB values). No FFT; the raw regional spectrum is never mutated."""
    return replace(
        regional,
        spectrum=spectrum_for_display(
            regional.spectrum, scale, show_filter_response=show_filter_response
        ),
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

    @staticmethod
    def regional_spectrum(
        section: SeismicSection,
        region: SectionRegion,
        dataset: SeismicDataset,
        job: Job,
        *,
        chunk_traces: int = REGION_SPECTRUM_CHUNK_TRACES,
    ) -> RegionalSpectrum:
        """Mean of per-trace amplitude spectra over the physically present
        traces of `region`. Reads only the section's rows for those traces
        (no I/O) in batches of `chunk_traces`, accumulating per-bin sums:
        temporary memory is chunk x bins, not region x bins."""
        if chunk_traces < 1:
            raise ValueError("chunk_traces must be at least 1")
        resolved = region.resolve(section)
        if not resolved.is_valid:
            raise RegionTooSmallError(
                f"region must contain at least {MIN_REGION_TRACES} trace, "
                f"found {resolved.n_present}"
            )
        rows = list(resolved.present_positions)
        n_bins = section.n_samples // 2 + 1
        sum_original = np.zeros(n_bins, dtype=np.float64)
        sum_filtered = np.zeros(n_bins, dtype=np.float64)
        for start in range(0, len(rows), chunk_traces):
            batch = rows[start : start + chunk_traces]
            sum_original += np.abs(np.fft.rfft(section.original[batch], axis=-1)).sum(
                axis=0
            )
            sum_filtered += np.abs(np.fft.rfft(section.filtered[batch], axis=-1)).sum(
                axis=0
            )
        count = len(rows)
        waveform = None
        if resolved.is_single_trace:
            row = rows[0]
            waveform = TraceWaveform(
                trace_index=resolved.trace_indices[0],
                coordinate=int(section.coordinates[row]),
                time_ms=section.time_ms,
                original=np.array(section.original[row]),
                filtered=np.array(section.filtered[row]),
            )
        return RegionalSpectrum(
            spectrum=_linear_spectrum(
                sum_original / count,
                sum_filtered / count,
                n_samples=section.n_samples,
                dataset=dataset,
                job=job,
            ),
            region=resolved,
            dataset=dataset,
            job=job,
            waveform=waveform,
        )


def _linear_spectrum(
    original: np.ndarray,
    filtered: np.ndarray,
    *,
    n_samples: int,
    dataset: SeismicDataset,
    job: Job,
) -> TraceSpectrum:
    """The rfft frequency axis and ONE theoretical zero-phase response
    from the job's SOS (|H|^2), whatever the number of traces behind the
    magnitudes."""
    fs_hz = 1000 / dataset.sample_rate_ms
    frequencies = np.fft.rfftfreq(n_samples, d=1 / fs_hz)
    sos = butterworth_sos(
        job.cutoff_hz,
        job.order,
        dataset.sample_rate_ms,
        filter_type=job.filter_type,
        upper_cutoff_hz=job.upper_cutoff_hz,
    )
    _, response = sosfreqz(sos, worN=frequencies, fs=fs_hz)
    return TraceSpectrum(
        frequencies_hz=frequencies,
        original=original,
        filtered=filtered,
        nyquist_hz=fs_hz / 2,
        cutoff_hz=job.cutoff_hz,
        filter_type=job.filter_type,
        upper_cutoff_hz=job.upper_cutoff_hz,
        filter_response=np.abs(response) ** 2,
    )
