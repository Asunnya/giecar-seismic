from giecar_seismic.domain.job import Job, JobStatus


def test_new_job_starts_in_created_status():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)

    assert job.status is JobStatus.CREATED


def test_start_moves_job_from_created_to_running():
    job = Job(dataset_id=1, cutoff_hz=30.0, order=4)

    job.start()

    assert job.status is JobStatus.RUNNING
