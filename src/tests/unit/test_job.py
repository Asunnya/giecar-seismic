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
