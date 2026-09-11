from dataclasses import dataclass
from enum import Enum, auto


class JobStatus(Enum):
    CREATED = auto()
    RUNNING = auto()


@dataclass
class Job:
    dataset_id: int
    cutoff_hz: float
    order: int
    status: JobStatus = JobStatus.CREATED

    def start(self) -> None:
        self.status = JobStatus.RUNNING
