from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from giecar_seismic.application.filter_jobs import DatasetRepository, JobRepository
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation, TraceGeometry
from giecar_seismic.domain.job import Job, JobStatus

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


@dataclass(frozen=True)
class TraceSpectrum:
    frequencies_hz: np.ndarray  # 0 .. nyquist
    original: np.ndarray  # |rfft|
    filtered: np.ndarray
    nyquist_hz: float
    cutoff_hz: float


@dataclass(frozen=True)
class ViewerContext:
    dataset: SeismicDataset
    job: Job


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

    def context(self, job_id: int) -> ViewerContext:
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

    def line_numbers(self, dataset_id: int, orientation: LineOrientation) -> list[int]:
        if orientation is LineOrientation.INLINE:
            return self._geometry.inline_numbers(dataset_id)
        return self._geometry.crossline_numbers(dataset_id)

    def load_section(
        self, job_id: int, orientation: LineOrientation, line_number: int
    ) -> SeismicSection:
        ctx = self.context(job_id)
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
        self, job_id: int, section: SeismicSection, coordinate: float
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
        ctx = self.context(job_id)
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
        return TraceSpectrum(
            frequencies_hz=frequencies,
            original=np.abs(np.fft.rfft(view.original)),
            filtered=np.abs(np.fft.rfft(view.filtered)),
            nyquist_hz=fs_hz / 2,
            cutoff_hz=view.job.cutoff_hz,
        )
