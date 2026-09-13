from collections.abc import Callable

from PyQt5.QtCore import QObject, pyqtSignal

from giecar_seismic.application.seismic_viewer import (
    SectionRegion,
    SeismicSection,
    SeismicViewerService,
    ViewerTarget,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.domain.job import Job

GeometryIndexBuilder = Callable[[SeismicDataset], bool]


class GeometryIndexWorker(QObject):
    """Runs the (bounded, batch-wise) geometry index build off the GUI
    thread. Emits finished(built: bool) or failed(message). Knows nothing
    about widgets, SQLAlchemy or segyio -- only the injected callable."""

    finished = pyqtSignal(bool)
    failed = pyqtSignal(str)
    terminal_signal_names = ("finished", "failed")

    def __init__(
        self, build_index: GeometryIndexBuilder, dataset: SeismicDataset
    ) -> None:
        super().__init__()
        self._build_index = build_index
        self._dataset = dataset

    def run(self) -> None:
        try:
            built = self._build_index(self._dataset)
        except Exception as exc:  # noqa: BLE001 -- must reach the GUI as a signal
            self.failed.emit(str(exc))
            return
        self.finished.emit(built)


class SectionLoadWorker(QObject):
    """Loads one inline/crossline (geometry query + selective SEG-Y and
    HDF5 reads, or the in-memory preview filter of that line) off the GUI
    thread and hands back a SeismicSection -- a small value object sized
    to that one line, never the volume."""

    terminal_signal_names = ("loaded", "failed")

    loaded = pyqtSignal(object)  # SeismicSection
    failed = pyqtSignal(str)

    def __init__(
        self,
        service: SeismicViewerService,
        target: ViewerTarget,
        orientation: LineOrientation,
        line_number: int,
    ) -> None:
        super().__init__()
        self._service = service
        self._target = target
        self._orientation = orientation
        self._line_number = line_number

    def run(self) -> None:
        try:
            section = self._service.load_section(
                self._target, self._orientation, self._line_number
            )
        except Exception as exc:  # noqa: BLE001 -- must reach the GUI as a signal
            self.failed.emit(str(exc))
            return
        self.loaded.emit(section)


class RegionalSpectrumWorker(QObject):
    """Aggregates the regional spectrum of an already-loaded section off
    the GUI thread: chunked FFT magnitudes over the region's present
    traces, no I/O of any kind. Emits computed(RegionalSpectrum) or
    failed(message); a region may hold hundreds of traces, so this must
    not run on the GUI thread."""

    computed = pyqtSignal(object)  # RegionalSpectrum
    failed = pyqtSignal(str)
    terminal_signal_names = ("computed", "failed")

    def __init__(
        self,
        service: SeismicViewerService,
        section: SeismicSection,
        region: SectionRegion,
        dataset: SeismicDataset,
        job: Job,
    ) -> None:
        super().__init__()
        self._service = service
        self._section = section
        self._region = region
        self._dataset = dataset
        self._job = job

    def run(self) -> None:
        try:
            regional = self._service.regional_spectrum(
                self._section, self._region, self._dataset, self._job
            )
        except Exception as exc:  # noqa: BLE001 -- must reach the GUI as a signal
            self.failed.emit(str(exc))
            return
        self.computed.emit(regional)
