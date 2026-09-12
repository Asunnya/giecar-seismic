from pathlib import Path

import numpy as np
import pytest
import segyio

from giecar_seismic.infrastructure.segy.dataset_importer import import_segy_dataset


@pytest.fixture
def small_survey_path(tmp_path: Path) -> Path:
    path = tmp_path / "survey.segy"

    # A tiny, deliberately irregular 3x3 inline/crossline grid (one position
    # missing) so the importer can't assume n_inlines * n_crosslines ==
    # trace_count -- same irregular-footprint situation the exploration
    # notebook found in the real dataset.
    inlines = [10, 10, 10, 11, 11, 12, 12, 12]
    crosslines = [1, 2, 3, 1, 2, 1, 2, 3]
    n_samples = 5

    spec = segyio.spec()
    spec.samples = list(range(n_samples))
    spec.tracecount = len(inlines)
    spec.format = 5  # IEEE float32

    with segyio.create(str(path), spec) as segy:
        for i, (inline, crossline) in enumerate(zip(inlines, crosslines)):
            segy.trace[i] = np.zeros(n_samples, dtype=np.float32)
            segy.header[i][segyio.TraceField.INLINE_3D] = inline
            segy.header[i][segyio.TraceField.CROSSLINE_3D] = crossline
        segy.bin[segyio.BinField.Interval] = 4000  # 4 ms

    return path


def test_import_segy_dataset_reads_metadata_without_loading_traces(small_survey_path):
    dataset = import_segy_dataset(small_survey_path, name="small survey")

    assert dataset.name == "small survey"
    assert dataset.source_path == str(small_survey_path)
    assert dataset.n_samples == 5
    assert dataset.sample_rate_ms == 4.0
    assert dataset.n_inlines == 3
    assert dataset.n_crosslines == 3
    assert dataset.n_traces == 8
    assert dataset.id is None


def test_import_segy_dataset_nyquist_matches_sample_rate(small_survey_path):
    dataset = import_segy_dataset(small_survey_path, name="small survey")

    assert dataset.nyquist_hz == 125.0


def test_import_segy_dataset_n_traces_is_the_physical_count_not_the_grid_product(
    small_survey_path,
):
    # 3 inlines x 3 crosslines = 9 geometric positions, but the fixture
    # only has 8 physical traces (one position is missing). The real
    # Volve survey has the same shape of problem: 401 x 720 = 288720
    # positions vs 288694 physical traces. n_traces must come from the
    # file's actual trace count, never from n_inlines * n_crosslines.
    dataset = import_segy_dataset(small_survey_path, name="small survey")

    assert dataset.n_traces == 8
    assert dataset.n_inlines * dataset.n_crosslines == 9
    assert dataset.n_traces != dataset.n_inlines * dataset.n_crosslines
