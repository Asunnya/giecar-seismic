import h5py
import numpy as np
import pytest
from sqlalchemy import text

from giecar_seismic.application.butterworth_filter import apply_butterworth_filter
from giecar_seismic.application.filter_jobs import CooperativeCancelToken
from giecar_seismic.domain.job import JobStatus
from tests.e2e.test_end_to_end_filter_pipeline import FOOTPRINTS, compose, write_segy
from tests.unit.test_filter_types import CASES


@pytest.mark.parametrize("kind,cutoff,upper,_passes", CASES)
def test_filter_types_stream_persist_and_reopen(tmp_path, kind, cutoff, upper, _passes):
    segy_path = tmp_path / "survey.segy"
    amplitudes = write_segy(segy_path, FOOTPRINTS[0], 128, 4000)
    db_path = tmp_path / "jobs.sqlite"
    engine, importer, service = compose(db_path, tmp_path / "outputs", chunk_size=3)
    dataset = importer(str(segy_path), "survey")
    job = service.create_filter_job(
        dataset.id, cutoff, 4, filter_type=kind, upper_cutoff_hz=upper
    )
    progress = []
    finished = service.run_filter_job(job.id, progress.append, CooperativeCancelToken())
    assert finished.status is JobStatus.COMPLETED
    assert progress == [38, 75, 100]
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT filter_type FROM jobs")).scalar_one()
            == kind.name
        )
    engine.dispose()
    fresh_engine, _, fresh_service = compose(db_path, tmp_path / "outputs", 3)
    try:
        reloaded = fresh_service.get_job_status(job.id)
        assert reloaded.filter_type is kind
        assert reloaded.upper_cutoff_hz == upper
        assert reloaded.status is JobStatus.COMPLETED
        with h5py.File(reloaded.output_path, "r") as output:
            assert output.attrs["complete"]
            expected = apply_butterworth_filter(
                amplitudes, cutoff, 4, 4.0, filter_type=kind, upper_cutoff_hz=upper
            )
            np.testing.assert_allclose(
                output["traces"][:], expected, rtol=1e-5, atol=1e-6
            )
    finally:
        fresh_engine.dispose()
