from collections.abc import Callable

from PyQt5.QtCore import QObject, pyqtSignal

from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus
from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset

DatasetImporter = Callable[[str, str], SeismicDataset]


class FilterJobWorker(QObject):
    """Runs FilterJobService.run_filter_job() off the GUI thread.

    Every dependency is received by construction, so this can be tested
    with plain Python fakes -- no QApplication, no real SEG-Y/HDF5, and no
    QThread required. In production it is meant to be moveToThread()'d
    onto a QThread before that thread's `started` signal invokes run();
    it talks back to whoever is listening (the GUI thread) exclusively
    through signals and never touches a QWidget, directly or indirectly.
    """

    progress = pyqtSignal(float)
    completed = pyqtSignal(object)  # Job, status == COMPLETED
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(object)  # Job, status == CANCELLED

    def __init__(
        self,
        service: FilterJobService,
        job_id: int,
        cancel_token: CooperativeCancelToken,
    ) -> None:
        super().__init__()
        self._service = service
        self._job_id = job_id
        self._cancel_token = cancel_token

    def run(self) -> None:
        try:
            job = self._service.run_filter_job(
                self._job_id,
                progress_callback=self.progress.emit,
                cancel_token=self._cancel_token,
            )
        except Exception as exc:  # noqa: BLE001 -- any failure must reach the GUI via a signal, never crash this thread silently
            self.failed.emit(str(exc))
            return

        self._emit_terminal_signal(job)

    def _emit_terminal_signal(self, job: Job) -> None:
        if job.status is JobStatus.COMPLETED:
            self.completed.emit(job)
        elif job.status is JobStatus.CANCELLED:
            self.cancelled.emit(job)
        else:
            self.failed.emit(job.error_message or f"job ended in status {job.status}")


class SegyImportWorker(QObject):
    """Runs import_segy_dataset() off the GUI thread.

    Reading SEG-Y trace headers (INLINE_3D/CROSSLINE_3D across every
    trace) is I/O that scales with trace count, not sample count -- it
    must not block the GUI thread even though it never touches trace
    amplitudes. The SEG-Y path and dataset name are received by
    construction; the importer function itself is injected too (default:
    the real import_segy_dataset), so tests can supply a fake importer
    without touching a real file. Never touches a QWidget -- talks back
    exclusively through signals.
    """

    succeeded = pyqtSignal(object)  # SeismicDataset
    failed = pyqtSignal(str)

    def __init__(
        self,
        source_path: str,
        name: str,
        importer: DatasetImporter = import_segy_dataset,
    ) -> None:
        super().__init__()
        self._source_path = source_path
        self._name = name
        self._importer = importer

    def run(self) -> None:
        try:
            dataset = self._importer(self._source_path, self._name)
        except Exception as exc:  # noqa: BLE001 -- any failure must reach the GUI via a signal, never crash this thread silently
            self.failed.emit(str(exc))
            return

        self.succeeded.emit(dataset)
