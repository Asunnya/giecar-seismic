from collections.abc import Callable

from PyQt5.QtCore import QObject, pyqtSignal

from giecar_seismic.application.seismic_viewer import SeismicViewerService
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import LineOrientation

GeometryIndexBuilder = Callable[[SeismicDataset], bool]


class GeometryIndexWorker(QObject):
    """Runs the (bounded, batch-wise) geometry index build off the GUI
    thread. Emits finished(built: bool) or failed(message). Knows nothing
    about widgets, SQLAlchemy or segyio -- only the injected callable."""

    finished = pyqtSignal(bool)
    failed = pyqtSignal(str)

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
    HDF5 reads) off the GUI thread and hands back a SeismicSection -- a
    small value object sized to that one line, never the volume."""

    loaded = pyqtSignal(object)  # SeismicSection
    failed = pyqtSignal(str)

    def __init__(
        self,
        service: SeismicViewerService,
        job_id: int,
        orientation: LineOrientation,
        line_number: int,
    ) -> None:
        super().__init__()
        self._service = service
        self._job_id = job_id
        self._orientation = orientation
        self._line_number = line_number

    def run(self) -> None:
        try:
            section = self._service.load_section(
                self._job_id, self._orientation, self._line_number
            )
        except Exception as exc:  # noqa: BLE001 -- must reach the GUI as a signal
            self.failed.emit(str(exc))
            return
        self.loaded.emit(section)
