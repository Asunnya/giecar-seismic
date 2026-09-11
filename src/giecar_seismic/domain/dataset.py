from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SeismicDataset:
    id: int
    name: str
    source_path: str
    n_inlines: int
    n_crosslines: int
    n_samples: int
    sample_rate_ms: float
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def nyquist_hz(self) -> float:
        sampling_frequency_hz = 1000 / self.sample_rate_ms
        return sampling_frequency_hz / 2
