from pathlib import Path

import segyio

from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.infrastructure.segy.reader import _iter_inline_crossline_batches

DEFAULT_IMPORT_HEADER_BATCH_SIZE = 4096


def import_segy_dataset(source_path: str | Path, name: str) -> SeismicDataset:
    """Builds a SeismicDataset by reading trace headers only.

    Never touches trace amplitudes. INLINE_3D/CROSSLINE_3D are read through
    bounded temporary arrays, while sets retain only distinct geometric
    identifiers. Working memory is therefore O(batch_size + unique inlines +
    unique crosslines), with no per-trace collection retained across batches.
    """
    with segyio.open(source_path, mode="r", ignore_geometry=True) as segy:
        n_samples = len(segy.samples)
        n_traces = segy.tracecount
        sample_rate_ms = float(segyio.tools.dt(segy)) / 1000

        unique_inlines: set[int] = set()
        unique_crosslines: set[int] = set()
        for _, batch_inlines, batch_crosslines in _iter_inline_crossline_batches(
            segy, DEFAULT_IMPORT_HEADER_BATCH_SIZE
        ):
            unique_inlines.update(int(value) for value in batch_inlines)
            unique_crosslines.update(int(value) for value in batch_crosslines)

    return SeismicDataset(
        name=name,
        source_path=str(source_path),
        n_inlines=len(unique_inlines),
        n_crosslines=len(unique_crosslines),
        n_traces=n_traces,
        n_samples=n_samples,
        sample_rate_ms=sample_rate_ms,
    )
