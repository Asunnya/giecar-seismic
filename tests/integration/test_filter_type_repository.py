from giecar_seismic.domain.job import FilterType
from giecar_seismic.infrastructure.database.engine import make_session_factory
from giecar_seismic.infrastructure.database.repositories import SqlAlchemyJobRepository
from tests.e2e.test_end_to_end_filter_pipeline import FOOTPRINTS, compose, write_segy


def test_repository_update_and_list_preserve_changed_filter_configuration(tmp_path):
    segy_path = tmp_path / "survey.segy"
    write_segy(segy_path, FOOTPRINTS[0], 64, 4000)
    engine, importer, service = compose(tmp_path / "jobs.sqlite", tmp_path / "out", 3)
    try:
        dataset = importer(str(segy_path), "survey")
        job = service.create_filter_job(dataset.id, 30, 4)
        jobs = SqlAlchemyJobRepository(make_session_factory(engine))
        job.filter_type = FilterType.BAND_PASS
        job.cutoff_hz, job.upper_cutoff_hz = 10, 40
        jobs.update(job)
        reloaded = jobs.list()[0]
        assert reloaded.filter_type is FilterType.BAND_PASS
        assert (reloaded.cutoff_hz, reloaded.upper_cutoff_hz) == (10, 40)
        job.filter_type = FilterType.HIGH_PASS
        job.upper_cutoff_hz = None
        jobs.update(job)
        assert jobs.get(job.id).upper_cutoff_hz is None
        assert jobs.get(job.id).filter_type is FilterType.HIGH_PASS
    finally:
        engine.dispose()
