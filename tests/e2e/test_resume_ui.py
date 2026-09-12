from pathlib import Path

import pytest
from PyQt5.QtCore import QThread

from giecar_seismic.application.filter_jobs import CooperativeCancelToken
from giecar_seismic.domain.job import JobStatus
from giecar_seismic.ui.main_window import MainWindow
from giecar_seismic.ui.workers import FilterJobWorker
from tests.e2e.test_resume_filter_job import resumed_service, setup_cancelled


@pytest.mark.parametrize(
    "status,exists,count,eligible",
    [
        (JobStatus.CANCELLED, True, 2, True),
        (JobStatus.CANCELLED, False, 2, False),
        (JobStatus.CANCELLED, True, 8, False),
        (JobStatus.FAILED, True, 2, False),
        (JobStatus.COMPLETED, True, 8, False),
        (JobStatus.CREATED, True, 0, False),
        (JobStatus.RUNNING, True, 2, False),
    ],
)
def test_resume_button_basic_eligibility(
    qapp, tmp_path, status, exists, count, eligible
):
    engine, dataset, job, _ = setup_cancelled(tmp_path)
    service, _ = resumed_service(engine, [])
    window = MainWindow()
    window._service = service
    window._dataset_trace_counts = {dataset.id: dataset.n_traces}
    job.status, job.processed_traces = status, count
    if not exists:
        Path(job.output_path).unlink()
    window._on_history_loaded([job], {dataset.id: "survey"})
    window._jobs_table.selectRow(0)
    assert window._resume_button.isEnabled() is eligible
    window.close()
    engine.dispose()


def test_resume_worker_uses_explicit_service_operation():
    calls = []

    class Service:
        def resume_filter_job(self, job_id, progress_callback, cancel_token):
            from giecar_seismic.domain.job import Job

            calls.append(job_id)
            progress_callback(75)
            return Job(1, 30, 4, status=JobStatus.COMPLETED)

    worker = FilterJobWorker(Service(), 7, CooperativeCancelToken(), resume=True)
    progresses, completed = [], []
    worker.progress.connect(progresses.append)
    worker.completed.connect(completed.append)
    worker.run()
    assert calls == [7]
    assert progresses == [75]
    assert len(completed) == 1


def test_resume_click_keeps_progress_and_runs_off_gui_thread(
    qapp, wait_for_signal, tmp_path, monkeypatch
):
    engine, _, _, _ = setup_cancelled(tmp_path)
    service, _ = resumed_service(engine, [])
    window = MainWindow(service=service)
    wait_for_signal(window.history_refreshed)
    window._jobs_table.selectRow(0)
    calls = []
    original = service.resume_filter_job

    def resume(*args, **kwargs):
        calls.append(QThread.currentThread() is qapp.thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "resume_filter_job", resume)
    window._resume_button.click()
    assert window._progress_bar.value() == 25
    assert not window._resume_button.isEnabled()
    assert not window._run_button.isEnabled()
    assert window._cancel_button.isEnabled()
    wait_for_signal(window.history_refreshed)
    assert calls == [False]
    assert window._table_jobs[0].status is JobStatus.COMPLETED
    if window._thread is not None:
        wait_for_signal(window._thread.finished)
    window.close()
    engine.dispose()
