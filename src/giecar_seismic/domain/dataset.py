from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SeismicDataset:
    name: str
    source_path: str
    n_inlines: int
    n_crosslines: int
    # Physical trace count read from the SEG-Y file itself. Deliberately
    # NOT derived from n_inlines * n_crosslines: a survey's inline/
    # crossline footprint can be irregular (the sample Volve file has
    # 401 x 720 = 288720 grid positions but only 288694 physical traces),
    # so the grid product over-counts. Streaming, progress and the HDF5
    # writer's expected_trace_count all need the real number.
    n_traces: int
    n_samples: int
    sample_rate_ms: float
    created_at: datetime = field(default_factory=datetime.now)
    id: int | None = None

    @property
    def nyquist_hz(self) -> float:
        sampling_frequency_hz = 1000 / self.sample_rate_ms
        return sampling_frequency_hz / 2
