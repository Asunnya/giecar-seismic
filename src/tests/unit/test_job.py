import pytest

from giecar_seismic.domain.job import InvalidTransitionError, Job, JobStatus


def test_new_job_starts_in_created_status():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)

    assert job.status is JobStatus.CREATED


def test_start_moves_job_from_created_to_running():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)

    job.start()

    assert job.status is JobStatus.RUNNING


def test_start_twice_raises_and_keeps_running_status():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)
    job.start()

    with pytest.raises(InvalidTransitionError):
        job.start()

    assert job.status is JobStatus.RUNNING


@pytest.mark.parametrize(
    "terminal_status",
    [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
)
def test_start_from_terminal_status_raises_and_keeps_status(terminal_status):
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4, status=terminal_status)

    with pytest.raises(InvalidTransitionError):
        job.start()

    assert job.status is terminal_status


def test_complete_moves_job_from_running_to_completed():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)
    job.start()

    job.complete()

    assert job.status is JobStatus.COMPLETED


@pytest.mark.parametrize(
    "non_running_status",
    [JobStatus.CREATED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
)
def test_complete_outside_running_raises_and_keeps_status(non_running_status):
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4, status=non_running_status)

    with pytest.raises(InvalidTransitionError):
        job.complete()

    assert job.status is non_running_status


def test_fail_moves_job_from_running_to_failed_and_records_error_message():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)
    job.start()

    job.fail("segyio read error")

    assert job.status is JobStatus.FAILED
    assert job.error_message == "segyio read error"


@pytest.mark.parametrize(
    "non_running_status",
    [JobStatus.CREATED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
)
def test_fail_outside_running_raises_and_keeps_status(non_running_status):
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4, status=non_running_status)

    with pytest.raises(InvalidTransitionError):
        job.fail("segyio read error")

    assert job.status is non_running_status
    assert job.error_message is None


def test_cancel_moves_job_from_running_to_cancelled():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)
    job.start()

    job.cancel()

    assert job.status is JobStatus.CANCELLED


@pytest.mark.parametrize(
    "non_running_status",
    [JobStatus.CREATED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
)
def test_cancel_outside_running_raises_and_keeps_status(non_running_status):
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4, status=non_running_status)

    with pytest.raises(InvalidTransitionError):
        job.cancel()

    assert job.status is non_running_status


# --- progress, output_path and lifecycle timestamps -----------------------


def _running_job() -> Job:
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)
    job.start()
    return job


def test_new_job_starts_with_zero_progress_no_output_and_only_created_at():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)

    assert job.progress == 0
    assert job.output_path is None
    assert job.created_at is not None
    assert job.started_at is None
    assert job.finished_at is None


def test_start_records_started_at_after_created_at():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)

    job.start()

    assert job.started_at is not None
    assert job.started_at >= job.created_at
    assert job.finished_at is None


def test_complete_records_finished_at_and_forces_progress_to_100():
    job = _running_job()
    job.advance_progress(40)

    job.complete()

    assert job.progress == 100
    assert job.finished_at is not None
    assert job.started_at is not None
    assert job.finished_at >= job.started_at


def test_fail_records_finished_at_and_preserves_the_progress_reached():
    job = _running_job()
    job.advance_progress(40)

    job.fail("disk full")

    assert job.progress == 40
    assert job.finished_at is not None
    assert job.started_at is not None
    assert job.finished_at >= job.started_at


def test_cancel_records_finished_at_and_preserves_the_progress_reached():
    job = _running_job()
    job.advance_progress(40)

    job.cancel()

    assert job.progress == 40
    assert job.finished_at is not None
    assert job.started_at is not None
    assert job.finished_at >= job.started_at


def test_advance_progress_accepts_monotonic_updates_while_running():
    job = _running_job()

    job.advance_progress(25)
    job.advance_progress(25)  # same value is not a regression
    job.advance_progress(75.5)

    assert job.progress == 75.5


def test_advance_progress_rejects_regression():
    job = _running_job()
    job.advance_progress(50)

    with pytest.raises(ValueError):
        job.advance_progress(49)

    assert job.progress == 50


@pytest.mark.parametrize("out_of_range", [-0.1, 100.1])
def test_advance_progress_rejects_values_outside_0_100(out_of_range):
    job = _running_job()

    with pytest.raises(ValueError):
        job.advance_progress(out_of_range)

    assert job.progress == 0


@pytest.mark.parametrize(
    "non_running_status",
    [JobStatus.CREATED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
)
def test_advance_progress_outside_running_raises_and_keeps_progress(
    non_running_status,
):
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4, status=non_running_status)

    with pytest.raises(InvalidTransitionError):
        job.advance_progress(10)

    assert job.progress == 0
