"""BuildGeometryIndexUseCase: bounded, batch-wise construction of the
(dataset, inline, crossline) -> physical trace index mapping, against an
in-memory fake repository. Persistence itself is covered in
test_sqlalchemy_geometry_repository.py.
"""

from collections.abc import Iterator

import pytest

from giecar_seismic.application.geometry_index import (
    BuildGeometryIndexUseCase,
    IncompleteGeometryIndexError,
)
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import TraceGeometry

# 3x3 envelope, position (11, 3) missing: 8 physical traces, not 9.
HEADERS = [(10, 1), (10, 2), (10, 3), (11, 1), (11, 2), (12, 1), (12, 2), (12, 3)]


def _dataset(n_traces: int = len(HEADERS)) -> SeismicDataset:
    return SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=3,
        n_crosslines=3,
        n_traces=n_traces,
        n_samples=64,
        sample_rate_ms=4.0,
    )


class FakeGeometryRepository:
    def __init__(self) -> None:
        self.rows: dict[int, list[TraceGeometry]] = {}
        self.batch_sizes: list[int] = []

    def count(self, dataset_id: int) -> int:
        return len(self.rows.get(dataset_id, []))

    def add_batch(self, dataset_id: int, batch: list[TraceGeometry]) -> None:
        self.batch_sizes.append(len(batch))
        self.rows.setdefault(dataset_id, []).extend(batch)

    def delete_for_dataset(self, dataset_id: int) -> None:
        self.rows.pop(dataset_id, None)


def _headers_in_batches(batch_size: int, headers=HEADERS):
    calls: list[tuple[str, int]] = []

    def read(
        source_path: str, requested_batch_size: int
    ) -> Iterator[list[TraceGeometry]]:
        calls.append((source_path, requested_batch_size))
        for start in range(0, len(headers), batch_size):
            yield [
                TraceGeometry(trace_index=i, inline=il, crossline=xl)
                for i, (il, xl) in enumerate(headers[start : start + batch_size], start)
            ]

    return read, calls


def test_index_is_built_batch_by_batch_never_holding_the_whole_survey():
    repository = FakeGeometryRepository()
    read, calls = _headers_in_batches(batch_size=3)
    use_case = BuildGeometryIndexUseCase(read, repository, batch_size=3)

    built = use_case(_dataset())

    assert built is True
    assert calls == [("/data/survey.segy", 3)]
    assert repository.batch_sizes == [3, 3, 2]  # memory ~ batch, not survey
    assert repository.count(1) == 8
    assert [(g.trace_index, g.inline, g.crossline) for g in repository.rows[1]] == [
        (i, il, xl) for i, (il, xl) in enumerate(HEADERS)
    ]


def test_index_does_not_assume_a_cartesian_product():
    repository = FakeGeometryRepository()
    read, _ = _headers_in_batches(batch_size=100)

    BuildGeometryIndexUseCase(read, repository)(_dataset())

    dataset = _dataset()
    assert repository.count(1) == dataset.n_traces
    assert repository.count(1) != dataset.n_inlines * dataset.n_crosslines
    assert (11, 3) not in {(g.inline, g.crossline) for g in repository.rows[1]}


def test_a_complete_index_is_detected_and_not_rebuilt():
    repository = FakeGeometryRepository()
    read, calls = _headers_in_batches(batch_size=100)
    use_case = BuildGeometryIndexUseCase(read, repository)
    use_case(_dataset())
    calls.clear()

    built_again = use_case(_dataset())

    assert built_again is False
    assert calls == []  # no header read at all
    assert repository.count(1) == 8


def test_an_incomplete_index_is_rebuilt_cleanly_from_scratch():
    # Policy: no resume. Anything short of n_traces rows (e.g. an
    # interrupted earlier build) is dropped and rebuilt in full, so the
    # index can never contain a mix of stale and fresh rows.
    repository = FakeGeometryRepository()
    repository.add_batch(1, [TraceGeometry(0, 999, 999), TraceGeometry(1, 999, 999)])
    read, calls = _headers_in_batches(batch_size=100)

    built = BuildGeometryIndexUseCase(read, repository)(_dataset())

    assert built is True
    assert len(calls) == 1
    assert repository.count(1) == 8
    assert all(g.inline != 999 for g in repository.rows[1])


def test_a_header_count_that_disagrees_with_the_dataset_is_an_error():
    repository = FakeGeometryRepository()
    read, _ = _headers_in_batches(batch_size=100)

    with pytest.raises(IncompleteGeometryIndexError):
        BuildGeometryIndexUseCase(read, repository)(_dataset(n_traces=9))

    # nothing half-built is left behind that could pass as complete
    assert repository.count(1) != 9


def test_dataset_without_id_cannot_be_indexed():
    dataset = _dataset()
    dataset.id = None
    read, _ = _headers_in_batches(batch_size=100)

    with pytest.raises(ValueError):
        BuildGeometryIndexUseCase(read, FakeGeometryRepository())(dataset)
