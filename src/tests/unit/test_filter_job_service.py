import pytest

from giecar_seismic.application.filter_jobs import (
    DatasetNotFoundError,
    FilterJobService,
    InvalidFilterParametersError,
    JobNotFoundError,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import InvalidTransitionError, Job, JobStatus


class FakeDatasetRepository:
    def __init__(self, datasets: list[SeismicDataset] | None = None):
        self._datasets = {dataset.id: dataset for dataset in datasets or []}

    def get(self, dataset_id: int) -> SeismicDataset | None:
        return self._datasets.get(dataset_id)


class FakeJobRepository:
    def __init__(self) -> None:
        self._jobs: dict[int, Job] = {}
        self._next_id = 1

    def add(self, job: Job) -> Job:
        job.id = self._next_id
        self._jobs[job.id] = job
        self._next_id += 1
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job: Job) -> None:
        assert job.id is not None
        self._jobs[job.id] = job

    def list(
        self, dataset_id: int | None = None, status: JobStatus | None = None
    ) -> list[Job]:
        return list(self._jobs.values())


@pytest.fixture
def dataset() -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=401,
        n_crosslines=720,
        n_samples=850,
        sample_rate_ms=4.0,  # nyquist_hz == 125.0
    )


@pytest.fixture
def service(dataset) -> FilterJobService:
    return FilterJobService(
        datasets=FakeDatasetRepository([dataset]),
        jobs=FakeJobRepository(),
    )


def test_create_filter_job_raises_when_dataset_does_not_exist():
    service = FilterJobService(
        datasets=FakeDatasetRepository([]),
        jobs=FakeJobRepository(),
    )

    with pytest.raises(DatasetNotFoundError):
        service.create_filter_job(dataset_id=1, cutoff_hz=30.0, order=4)


def test_create_filter_job_returns_persisted_created_job(service, dataset):
    job = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    assert job.status is JobStatus.CREATED
    assert job.dataset_id == dataset.id
    assert job.cutoff_hz == 30.0
    assert job.order == 4
    assert job.id is not None


@pytest.mark.parametrize("invalid_cutoff_hz", [0, -10.0, 125.0, 200.0])
def test_create_filter_job_rejects_cutoff_outside_nyquist_range(
    service, dataset, invalid_cutoff_hz
):
    with pytest.raises(InvalidFilterParametersError):
        service.create_filter_job(
            dataset_id=dataset.id, cutoff_hz=invalid_cutoff_hz, order=4
        )


@pytest.mark.parametrize("invalid_order", [1, 0, -1, 9, 20])
def test_create_filter_job_rejects_order_outside_accepted_range(
    service, dataset, invalid_order
):
    with pytest.raises(InvalidFilterParametersError):
        service.create_filter_job(
            dataset_id=dataset.id, cutoff_hz=30.0, order=invalid_order
        )


def test_create_filter_job_does_not_persist_job_on_validation_failure(
    service, dataset
):
    with pytest.raises(InvalidFilterParametersError):
        service.create_filter_job(dataset_id=dataset.id, cutoff_hz=-10.0, order=4)

    assert service.list_jobs() == []


def test_get_job_status_returns_the_job(service, dataset):
    created = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    fetched = service.get_job_status(created.id)

    assert fetched is created


def test_get_job_status_raises_when_job_does_not_exist(service):
    with pytest.raises(JobNotFoundError):
        service.get_job_status(999)


def test_cancel_job_raises_when_job_does_not_exist(service):
    with pytest.raises(JobNotFoundError):
        service.cancel_job(999)


def test_cancel_job_moves_running_job_to_cancelled_and_persists_it(service, dataset):
    created = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    created.start()

    service.cancel_job(created.id)

    assert service.get_job_status(created.id).status is JobStatus.CANCELLED


def test_cancel_job_on_non_running_job_raises_invalid_transition(service, dataset):
    created = service.create_filter_job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)

    with pytest.raises(InvalidTransitionError):
        service.cancel_job(created.id)

    assert service.get_job_status(created.id).status is JobStatus.CREATED
