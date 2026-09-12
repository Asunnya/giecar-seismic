from collections.abc import Callable, Iterable
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from giecar_seismic.application.filter_jobs import (
    CooperativeCancelToken,
    FilterJobService,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus

# Full import: read SEG-Y metadata *and* persist -- returns a
# SeismicDataset with a real id. In production this is
# application.dataset_import.ImportDatasetUseCase; tests pass plain
# functions. The worker deliberately doesn't know which.
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
    """Runs a dataset import (read SEG-Y metadata + persist) off the GUI
    thread.

    Reading SEG-Y trace headers (INLINE_3D/CROSSLINE_3D across every
    trace) is I/O that scales with trace count -- it must not block the
    GUI thread even though it never touches trace amplitudes; the
    SQLite write that follows must not either. The SEG-Y path, dataset
    name and the importer callable are all received by construction: the
    worker knows nothing about segyio, sessions, engines or ORM models,
    and tests supply a plain function. Never touches a QWidget -- talks
    back exclusively through signals.
    """

    succeeded = pyqtSignal(object)  # SeismicDataset
    failed = pyqtSignal(str)

    def __init__(
        self,
        source_path: str,
        name: str,
        importer: DatasetImporter,
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


def dataset_labels(
    datasets: dict[int, SeismicDataset], missing: Iterable[int] = ()
) -> dict[int, str]:
    """Display label per dataset id: the SEG-Y file name; ` (#id)` is
    appended only when two datasets share a file name. A dataset row
    that no longer exists falls back to `Dataset #id`."""
    names = {i: Path(d.source_path).name for i, d in datasets.items()}
    counts: dict[str, int] = {}
    for name in names.values():
        counts[name] = counts.get(name, 0) + 1
    labels = {
        i: name if counts[name] == 1 else f"{name} (#{i})" for i, name in names.items()
    }
    labels.update({i: f"Dataset #{i}" for i in missing})
    return labels


class JobHistoryWorker(QObject):
    """Runs FilterJobService.list_jobs(dataset_id, status) off the GUI
    thread and emits the resulting list of domain Jobs together with a
    display label per dataset id (file name) -- small metadata objects
    only. Never touches a QWidget."""

    succeeded = pyqtSignal(object, object)  # list[Job], dict[int, str]
    failed = pyqtSignal(str)

    def __init__(
        self,
        service: FilterJobService,
        dataset_id: int | None,
        status: JobStatus | None,
    ) -> None:
        super().__init__()
        self._service = service
        self._dataset_id = dataset_id
        self._status = status

    def run(self) -> None:
        try:
            jobs = self._service.list_jobs(
                dataset_id=self._dataset_id, status=self._status
            )
            ids = {job.dataset_id for job in jobs}
            found = {
                i: d for i in ids if (d := self._service.get_dataset(i)) is not None
            }
            labels = dataset_labels(found, missing=[i for i in ids if i not in found])
        except Exception as exc:  # noqa: BLE001 -- any failure must reach the GUI via a signal
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(jobs, labels)
