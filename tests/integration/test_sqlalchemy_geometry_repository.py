from pathlib import Path

import pytest
from sqlalchemy import inspect

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.geometry import TraceGeometry
from giecar_seismic.infrastructure.database.engine import (
    create_schema,
    create_sqlite_engine,
    make_session_factory,
)
from giecar_seismic.infrastructure.database.repositories import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyGeometryRepository,
)

# 3x3 envelope with (11, 3) missing; deliberately out of physical order
# so "ordered by crossline/inline" is a real sort, not insertion order.
HEADERS = [(12, 3), (10, 1), (11, 2), (10, 3), (12, 1), (10, 2), (11, 1), (12, 2)]


@pytest.fixture
def repo(tmp_path: Path):
    engine = create_sqlite_engine(tmp_path / "g.sqlite")
    create_schema(engine)
    factory = make_session_factory(engine)
    dataset = SqlAlchemyDatasetRepository(factory).add(
        SeismicDataset(
            name="s",
            source_path="/s.segy",
            n_inlines=3,
            n_crosslines=3,
            n_traces=8,
            n_samples=4,
            sample_rate_ms=4.0,
        )
    )
    assert dataset.id is not None
    geometry = SqlAlchemyGeometryRepository(factory)
    geometry.add_batch(
        dataset.id,
        [TraceGeometry(i, il, xl) for i, (il, xl) in enumerate(HEADERS[:5])],
    )
    geometry.add_batch(
        dataset.id,
        [TraceGeometry(i, il, xl) for i, (il, xl) in enumerate(HEADERS[5:], 5)],
    )
    return geometry, dataset.id, engine


def test_count_reflects_every_persisted_batch(repo):
    geometry, dataset_id, _ = repo
    assert geometry.count(dataset_id) == 8
    assert geometry.count(dataset_id + 1) == 0


def test_inline_query_returns_physical_indices_ordered_by_crossline(repo):
    geometry, dataset_id, _ = repo

    rows = geometry.traces_for_inline(dataset_id, 10)

    assert [(g.crossline, g.trace_index) for g in rows] == [(1, 1), (2, 5), (3, 3)]
    assert all(g.inline == 10 for g in rows)


def test_crossline_query_returns_physical_indices_ordered_by_inline(repo):
    geometry, dataset_id, _ = repo

    rows = geometry.traces_for_crossline(dataset_id, 3)

    # crossline 3 exists for inlines 10 and 12 only -- (11, 3) is missing
    assert [(g.inline, g.trace_index) for g in rows] == [(10, 3), (12, 0)]


def test_line_numbers_are_distinct_sorted_and_scoped_to_the_dataset(repo):
    geometry, dataset_id, _ = repo

    assert geometry.inline_numbers(dataset_id) == [10, 11, 12]
    assert geometry.crossline_numbers(dataset_id) == [1, 2, 3]
    assert geometry.inline_numbers(dataset_id + 1) == []


def test_trace_lookup_by_physical_index(repo):
    geometry, dataset_id, _ = repo

    assert geometry.get_trace(dataset_id, 6) == TraceGeometry(6, 11, 1)
    assert geometry.get_trace(dataset_id, 99) is None


def test_delete_for_dataset_removes_only_that_dataset(repo):
    geometry, dataset_id, _ = repo
    other = dataset_id + 1  # FK: needs a real dataset row
    factory = geometry._session_factory
    other_ds = SqlAlchemyDatasetRepository(factory).add(
        SeismicDataset(
            name="o",
            source_path="/o.segy",
            n_inlines=1,
            n_crosslines=1,
            n_traces=1,
            n_samples=4,
            sample_rate_ms=4.0,
        )
    )
    assert other_ds.id == other
    geometry.add_batch(other, [TraceGeometry(0, 1, 1)])

    geometry.delete_for_dataset(dataset_id)

    assert geometry.count(dataset_id) == 0
    assert geometry.count(other) == 1


def test_schema_has_lookup_indexes_for_inline_and_crossline_queries(repo):
    _, _, engine = repo
    indexed_columns = {
        tuple(index["column_names"])
        for index in inspect(engine).get_indexes("trace_geometry")
    }
    assert ("dataset_id", "inline_number") in indexed_columns
    assert ("dataset_id", "crossline_number") in indexed_columns
