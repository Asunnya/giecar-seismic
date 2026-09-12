from giecar_seismic.infrastructure.segy import dataset_importer


class RecordingAttribute:
    def __init__(self, values, requests):
        self._values = values
        self._requests = requests

    def __getitem__(self, requested):
        self._requests.append(requested)
        return self._values[requested]


class FakeSegy:
    def __init__(self, inlines, crosslines, n_samples):
        self.tracecount = len(inlines)
        self.samples = range(n_samples)
        self._values = {
            dataset_importer.segyio.TraceField.INLINE_3D: inlines,
            dataset_importer.segyio.TraceField.CROSSLINE_3D: crosslines,
        }
        self.requests = []
        self.amplitudes_accessed = False

    def attributes(self, field):
        return RecordingAttribute(self._values[field], self.requests)

    @property
    def trace(self):
        self.amplitudes_accessed = True
        raise AssertionError("metadata import must not access trace amplitudes")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None


def test_import_streams_header_columns_in_bounded_batches(monkeypatch):
    batch_size = dataset_importer.DEFAULT_IMPORT_HEADER_BATCH_SIZE
    trace_count = batch_size * 3 + 17
    # Repeated identifiers cross every batch boundary and physical order is
    # deliberately unrelated to sorted inline/crossline order.
    inlines = [100 + (index * 7) % 11 for index in range(trace_count)]
    crosslines = [200 + (index * 13) % 17 for index in range(trace_count)]
    fake = FakeSegy(inlines, crosslines, n_samples=37)
    opened = []

    def open_segy(path, *, mode, ignore_geometry):
        opened.append((path, mode, ignore_geometry))
        return fake

    monkeypatch.setattr(dataset_importer.segyio, "open", open_segy)
    monkeypatch.setattr(dataset_importer.segyio.tools, "dt", lambda _segy: 2500)

    result = dataset_importer.import_segy_dataset("irregular.sgy", "irregular")

    assert opened == [("irregular.sgy", "r", True)]
    assert result.n_traces == trace_count
    assert result.n_traces != result.n_inlines * result.n_crosslines
    assert result.n_inlines == 11
    assert result.n_crosslines == 17
    assert result.n_samples == 37
    assert result.sample_rate_ms == 2.5
    assert result.source_path == "irregular.sgy"
    assert fake.amplitudes_accessed is False

    assert len(fake.requests) == 2 * 4
    assert all(isinstance(request, slice) for request in fake.requests)
    assert all(request.start is not None for request in fake.requests)
    assert all(request.stop is not None for request in fake.requests)
    assert all(request.step is None for request in fake.requests)
    assert max(request.stop - request.start for request in fake.requests) <= batch_size
    assert [request for request in fake.requests if request.start == 0] == [
        slice(0, batch_size),
        slice(0, batch_size),
    ]
