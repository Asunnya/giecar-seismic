from pathlib import Path

import numpy as np
import segyio

from giecar_seismic.domain.dataset import SeismicDataset


def import_segy_dataset(source_path: str | Path, name: str) -> SeismicDataset:
    """Builds a SeismicDataset by reading trace headers only.

    Never touches trace amplitudes: n_inlines/n_crosslines come from the
    per-trace INLINE_3D/CROSSLINE_3D header attributes, which is O(n_traces)
    in header memory but independent of the sample volume. See the "scope
    boundary" note in notebooks/01_inspect_segy.ipynb -- this is the same
    trade-off, deliberately kept out of the streaming trace-reading path.
    """
    with segyio.open(source_path, mode="r", ignore_geometry=True) as segy:
        n_samples = len(segy.samples)
        n_traces = segy.tracecount
        sample_rate_ms = float(segyio.tools.dt(segy)) / 1000

        inline_values = np.asarray(segy.attributes(segyio.TraceField.INLINE_3D)[:])
        crossline_values = np.asarray(
            segy.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        )

    n_inlines = len(np.unique(inline_values))
    n_crosslines = len(np.unique(crossline_values))

    return SeismicDataset(
        name=name,
        source_path=str(source_path),
        n_inlines=n_inlines,
        n_crosslines=n_crosslines,
        n_traces=n_traces,
        n_samples=n_samples,
        sample_rate_ms=sample_rate_ms,
    )
