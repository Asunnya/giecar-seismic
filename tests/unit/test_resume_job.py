import pytest

from giecar_seismic.domain.job import InvalidTransitionError, Job, JobStatus


def test_resume_preserves_logical_timestamps_and_records_new_finish():
    job = Job(1, 30, 4)
    job.start()
    created, started = job.created_at, job.started_at
    for count in (1, 2):
        job.cancel()
        assert job.finished_at is not None
        job.resume()
        assert job.status is JobStatus.RUNNING
        assert job.resume_count == count
        assert (job.created_at, job.started_at) == (created, started)
        assert job.finished_at is None
    job.complete()
    assert job.finished_at is not None


@pytest.mark.parametrize(
    "status",
    [JobStatus.CREATED, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED],
)
def test_resume_rejects_every_other_status(status):
    job = Job(1, 30, 4, status=status)
    with pytest.raises(InvalidTransitionError):
        job.resume()
    assert job.resume_count == 0


@pytest.mark.parametrize("count", [-1, 11, 2, 3.5, True])
def test_checkpoint_rejects_invalid_or_regressing_counts(count):
    job = Job(1, 30, 4)
    job.start()
    job.record_checkpoint(3, 10)
    with pytest.raises(ValueError):
        job.record_checkpoint(count, 10)
    assert job.processed_traces == 3
    assert job.progress == 30
