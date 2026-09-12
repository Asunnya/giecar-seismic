from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import Job, JobStatus
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
)

# Naive on purpose: SeismicDataset.created_at defaults to datetime.now(),
# which is naive, so the persisted value must round-trip as naive too.
CREATED_AT = datetime(2026, 9, 11, 12, 30, 45)  # noqa: DTZ001


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "giecar.sqlite"


@pytest.fixture
def session_factory(db_path: Path):
    engine = create_sqlite_engine(db_path)
    create_schema(engine)
    return make_session_factory(engine)


@pytest.fixture
def datasets(session_factory) -> SqlAlchemyDatasetRepository:
    return SqlAlchemyDatasetRepository(session_factory)


@pytest.fixture
def jobs(session_factory) -> SqlAlchemyJobRepository:
    return SqlAlchemyJobRepository(session_factory)


def _dataset(name: str = "survey") -> SeismicDataset:
    return SeismicDataset(
        name=name,
        source_path=f"/data/{name}.segy",
        n_inlines=401,
        n_crosslines=720,
        n_traces=288694,  # < 401 * 720 == 288720: irregular footprint
        n_samples=850,
        sample_rate_ms=4.0,
        created_at=CREATED_AT,
    )


# --- DatasetRepository ----------------------------------------------------


def test_add_dataset_without_id_assigns_a_persistent_id(datasets):
    dataset = _dataset()
    assert dataset.id is None

    persisted = datasets.add(dataset)

    assert persisted.id is not None
    assert isinstance(persisted.id, int)


def test_get_dataset_preserves_every_field(datasets):
    persisted = datasets.add(_dataset())
    assert persisted.id is not None

    fetched = datasets.get(persisted.id)

    assert fetched is not None
    assert fetched.id == persisted.id
    assert fetched.name == "survey"
    assert fetched.source_path == "/data/survey.segy"
    assert fetched.n_inlines == 401
    assert fetched.n_crosslines == 720
    assert fetched.n_traces == 288694
    assert fetched.n_samples == 850
    assert fetched.sample_rate_ms == 4.0
    assert fetched.created_at == CREATED_AT
    assert fetched.nyquist_hz == 125.0


def test_get_missing_dataset_returns_none(datasets):
    assert datasets.get(999) is None


def test_returned_dataset_is_usable_after_the_session_is_closed(datasets):
    # add() opens and closes its own session; the returned entity must be
    # a plain domain object, not something lazily bound to that session.
    persisted = datasets.add(_dataset())

    assert persisted.name == "survey"
    assert persisted.nyquist_hz == 125.0
    assert persisted.created_at == CREATED_AT


def test_dataset_survives_reopening_the_repository_from_the_same_file(
    db_path, session_factory
):
    persisted = SqlAlchemyDatasetRepository(session_factory).add(_dataset())
    assert persisted.id is not None

    # a brand-new engine/session factory over the same file proves the
    # row actually reached SQLite, not just a Python object cache.
    reopened = SqlAlchemyDatasetRepository(
        make_session_factory(create_sqlite_engine(db_path))
    )
    fetched = reopened.get(persisted.id)

    assert fetched is not None
    assert fetched.name == "survey"


# --- JobRepository --------------------------------------------------------


def _persisted_dataset_id(datasets: SqlAlchemyDatasetRepository) -> int:
    dataset_id = datasets.add(_dataset()).id
    assert dataset_id is not None
    return dataset_id


def test_add_created_job_assigns_an_id(datasets, jobs):
    dataset_id = _persisted_dataset_id(datasets)
    job = Job(dataset_id=dataset_id, cutoff_hz=30.0, order=4)
    assert job.id is None

    persisted = jobs.add(job)

    assert persisted.id is not None
    assert persisted.status is JobStatus.CREATED


def test_get_job_preserves_parameters_and_status(datasets, jobs):
    dataset_id = _persisted_dataset_id(datasets)
    persisted = jobs.add(Job(dataset_id=dataset_id, cutoff_hz=30.0, order=4))
    assert persisted.id is not None

    fetched = jobs.get(persisted.id)

    assert fetched is not None
    assert fetched.id == persisted.id
    assert fetched.dataset_id == dataset_id
    assert fetched.cutoff_hz == 30.0
    assert fetched.order == 4
    assert fetched.status is JobStatus.CREATED
    assert fetched.error_message is None


def test_get_missing_job_returns_none(jobs):
    assert jobs.get(999) is None


@pytest.mark.parametrize(
    ("reach_status", "expected_status", "expected_error"),
    [
        (lambda job: job.start(), JobStatus.RUNNING, None),
        (lambda job: (job.start(), job.complete()), JobStatus.COMPLETED, None),
        (
            lambda job: (job.start(), job.fail("disk full")),
            JobStatus.FAILED,
            "disk full",
        ),
        (lambda job: (job.start(), job.cancel()), JobStatus.CANCELLED, None),
    ],
    ids=["running", "completed", "failed", "cancelled"],
)
def test_update_persists_status_transitions(
    datasets, jobs, reach_status, expected_status, expected_error
):
    dataset_id = _persisted_dataset_id(datasets)
    job = jobs.add(Job(dataset_id=dataset_id, cutoff_hz=30.0, order=4))
    assert job.id is not None

    reach_status(job)
    jobs.update(job)

    fetched = jobs.get(job.id)
    assert fetched is not None
    assert fetched.status is expected_status
    assert fetched.error_message == expected_error


def test_update_unknown_job_raises(jobs):
    with pytest.raises(LookupError):
        jobs.update(Job(dataset_id=1, cutoff_hz=30.0, order=4, id=999))


def test_update_job_without_id_raises(jobs):
    with pytest.raises(ValueError, match="id"):
        jobs.update(Job(dataset_id=1, cutoff_hz=30.0, order=4))


def test_list_without_filters_returns_every_job(datasets, jobs):
    dataset_id = _persisted_dataset_id(datasets)
    jobs.add(Job(dataset_id=dataset_id, cutoff_hz=30.0, order=4))
    jobs.add(Job(dataset_id=dataset_id, cutoff_hz=40.0, order=6))

    listed = jobs.list()

    assert sorted(job.cutoff_hz for job in listed) == [30.0, 40.0]


def test_list_filters_by_dataset_id(datasets, jobs):
    dataset_a = _persisted_dataset_id(datasets)
    dataset_b = datasets.add(_dataset("other")).id
    assert dataset_b is not None
    jobs.add(Job(dataset_id=dataset_a, cutoff_hz=30.0, order=4))
    jobs.add(Job(dataset_id=dataset_b, cutoff_hz=40.0, order=4))

    listed = jobs.list(dataset_id=dataset_b)

    assert [job.dataset_id for job in listed] == [dataset_b]
    assert [job.cutoff_hz for job in listed] == [40.0]


def test_list_filters_by_status(datasets, jobs):
    dataset_id = _persisted_dataset_id(datasets)
    jobs.add(Job(dataset_id=dataset_id, cutoff_hz=30.0, order=4))
    running = jobs.add(Job(dataset_id=dataset_id, cutoff_hz=40.0, order=4))
    running.start()
    jobs.update(running)

    listed = jobs.list(status=JobStatus.RUNNING)

    assert [job.cutoff_hz for job in listed] == [40.0]
    assert all(job.status is JobStatus.RUNNING for job in listed)


def test_list_combines_dataset_and_status_filters(datasets, jobs):
    dataset_a = _persisted_dataset_id(datasets)
    dataset_b = datasets.add(_dataset("other")).id
    assert dataset_b is not None
    jobs.add(Job(dataset_id=dataset_a, cutoff_hz=10.0, order=4))
    a_running = jobs.add(Job(dataset_id=dataset_a, cutoff_hz=20.0, order=4))
    b_running = jobs.add(Job(dataset_id=dataset_b, cutoff_hz=30.0, order=4))
    for job in (a_running, b_running):
        job.start()
        jobs.update(job)

    listed = jobs.list(dataset_id=dataset_a, status=JobStatus.RUNNING)

    assert [job.cutoff_hz for job in listed] == [20.0]


def test_job_foreign_key_to_dataset_is_enforced(jobs):
    # SQLite only enforces FOREIGN KEY constraints when
    # PRAGMA foreign_keys=ON is issued per connection -- declaring the
    # FK in the schema alone is not enough. This proves the engine
    # configuration actually turns it on.
    with pytest.raises(IntegrityError):
        jobs.add(Job(dataset_id=999, cutoff_hz=30.0, order=4))


# --- Integration: real persistence across sessions/repositories -----------


def test_dataset_and_job_survive_fresh_repositories_over_the_same_database(db_path):
    engine = create_sqlite_engine(db_path)
    create_schema(engine)
    first_factory = make_session_factory(engine)

    dataset = SqlAlchemyDatasetRepository(first_factory).add(_dataset())
    assert dataset.id is not None
    job = SqlAlchemyJobRepository(first_factory).add(
        Job(dataset_id=dataset.id, cutoff_hz=30.0, order=4)
    )
    assert job.id is not None
    job.start()
    SqlAlchemyJobRepository(first_factory).update(job)
    engine.dispose()

    # Everything above is gone: new engine, new session factory, new
    # repository instances. Whatever comes back now came from the file.
    fresh_factory = make_session_factory(create_sqlite_engine(db_path))
    fetched_dataset = SqlAlchemyDatasetRepository(fresh_factory).get(dataset.id)
    fetched_job = SqlAlchemyJobRepository(fresh_factory).get(job.id)

    assert fetched_dataset is not None
    assert fetched_dataset.name == "survey"
    assert fetched_dataset.nyquist_hz == 125.0
    assert fetched_dataset.n_traces == 288694
    assert (
        fetched_dataset.n_traces
        != fetched_dataset.n_inlines * fetched_dataset.n_crosslines
    )
    assert fetched_job is not None
    assert fetched_job.dataset_id == fetched_dataset.id
    assert fetched_job.status is JobStatus.RUNNING
    assert fetched_job.cutoff_hz == 30.0
    assert fetched_job.order == 4
