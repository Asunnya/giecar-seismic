import pytest

from giecar_seismic.__main__ import parse_filter_process_count


@pytest.mark.parametrize(("raw", "expected"), [(None, 1), ("", 1), ("1", 1), ("4", 4)])
def test_filter_process_count_parsing(raw, expected):
    assert parse_filter_process_count(raw, available_cpus=8) == expected


@pytest.mark.parametrize("raw", ["zero", "1.5", "0", "-2", " true "])
def test_filter_process_count_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="GIECAR_FILTER_PROCESSES"):
        parse_filter_process_count(raw, available_cpus=8)


def test_filter_process_count_rejects_oversubscription():
    with pytest.raises(ValueError, match="available CPUs.*4"):
        parse_filter_process_count("5", available_cpus=4)
