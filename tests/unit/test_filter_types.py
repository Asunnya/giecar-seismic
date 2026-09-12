import numpy as np
import pytest

from giecar_seismic.application.butterworth_filter import apply_butterworth_filter
from giecar_seismic.application.filter_jobs import (
    FilterJobService,
    InvalidFilterParametersError,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import FilterType, Job
from tests.unit.test_filter_job_service import (
    FakeDatasetRepository,
    FakeJobRepository,
)


@pytest.fixture
def dataset():
    return SeismicDataset("survey", "/survey.segy", 1, 1, 4, 128, 4.0, id=1)


CASES = [
    (FilterType.LOW_PASS, 50.0, None, (True, True, False)),
    (FilterType.HIGH_PASS, 15.0, None, (False, True, True)),
    (FilterType.BAND_PASS, 15.0, 50.0, (False, True, False)),
]


@pytest.mark.parametrize("kind,cutoff,upper,passes", CASES)
def test_three_frequency_components_and_independent_traces(kind, cutoff, upper, passes):
    t = np.arange(4000) / 250
    components = np.sin(2 * np.pi * np.array([3, 25, 90])[:, None] * t)
    signal = components.sum(axis=0)
    chunk = np.stack([signal, 2 * signal, np.zeros_like(signal)])
    filtered = apply_butterworth_filter(
        chunk, cutoff, 4, 4.0, filter_type=kind, upper_cutoff_hz=upper
    )
    middle = slice(1000, 3000)
    for component, passes_band in zip(components, passes, strict=True):
        amplitude = 2 * np.mean(filtered[0, middle] * component[middle])
        assert amplitude > 0.95 if passes_band else abs(amplitude) < 0.05
    np.testing.assert_allclose(filtered[1], 2 * filtered[0])
    np.testing.assert_array_equal(filtered[2], 0)
    assert filtered.shape == chunk.shape


def test_legacy_job_and_three_argument_contract_default_to_lowpass(dataset):
    service = FilterJobService(FakeDatasetRepository([dataset]), FakeJobRepository())
    for job in (Job(1, 30.0, 4), service.create_filter_job(1, 30.0, 4)):
        assert job.filter_type is FilterType.LOW_PASS
        assert job.upper_cutoff_hz is None


@pytest.mark.parametrize("kind", list(FilterType))
@pytest.mark.parametrize("cutoff", [0, -1, 125, 150, float("nan")])
def test_invalid_primary_cutoffs(dataset, kind, cutoff):
    service = FilterJobService(FakeDatasetRepository([dataset]), FakeJobRepository())
    with pytest.raises(InvalidFilterParametersError, match="cutoff"):
        service.create_filter_job(1, cutoff, 4, filter_type=kind)
    assert service.list_jobs() == []


@pytest.mark.parametrize("upper", [None, 10, 9, 125, 150, float("nan")])
def test_invalid_bandpass_upper_cutoff(dataset, upper):
    service = FilterJobService(FakeDatasetRepository([dataset]), FakeJobRepository())
    with pytest.raises(InvalidFilterParametersError, match="upper_cutoff_hz"):
        service.create_filter_job(
            1, 10, 4, filter_type=FilterType.BAND_PASS, upper_cutoff_hz=upper
        )
    assert service.list_jobs() == []


@pytest.mark.parametrize("kind", [FilterType.LOW_PASS, FilterType.HIGH_PASS])
def test_unexpected_upper_cutoff_rejected(dataset, kind):
    service = FilterJobService(FakeDatasetRepository([dataset]), FakeJobRepository())
    with pytest.raises(InvalidFilterParametersError, match="upper_cutoff_hz"):
        service.create_filter_job(1, 10, 4, filter_type=kind, upper_cutoff_hz=40)


@pytest.mark.parametrize("kind", list(FilterType))
@pytest.mark.parametrize("order", [0, 1, 9, 2.5, True])
def test_invalid_orders(dataset, kind, order):
    service = FilterJobService(FakeDatasetRepository([dataset]), FakeJobRepository())
    with pytest.raises(InvalidFilterParametersError, match="order"):
        service.create_filter_job(
            1,
            10,
            order,
            filter_type=kind,
            upper_cutoff_hz=40 if kind is FilterType.BAND_PASS else None,
        )


@pytest.mark.parametrize("kind", list(FilterType))
@pytest.mark.parametrize("order", [2, 8])
def test_order_boundaries_are_accepted(dataset, kind, order):
    service = FilterJobService(FakeDatasetRepository([dataset]), FakeJobRepository())
    job = service.create_filter_job(
        1,
        10,
        order,
        filter_type=kind,
        upper_cutoff_hz=40 if kind is FilterType.BAND_PASS else None,
    )
    assert job.order == order
    assert job.filter_type is kind


def test_invalid_filter_type_is_rejected_before_persistence(dataset):
    service = FilterJobService(FakeDatasetRepository([dataset]), FakeJobRepository())
    with pytest.raises(InvalidFilterParametersError, match="filter_type"):
        service.create_filter_job(1, 10, 4, filter_type="high cut")
    assert service.list_jobs() == []
