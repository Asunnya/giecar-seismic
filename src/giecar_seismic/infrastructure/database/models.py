from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DatasetModel(Base):
    """SQLAlchemy row for SeismicDataset.

    Deliberately a separate class from the domain dataclass: the domain
    stays free of SQLAlchemy, and the ORM stays free of domain behaviour
    (nyquist_hz, etc.). Conversion between the two is explicit -- see
    repositories.py.
    """

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    n_inlines: Mapped[int] = mapped_column(Integer, nullable=False)
    n_crosslines: Mapped[int] = mapped_column(Integer, nullable=False)
    n_traces: Mapped[int] = mapped_column(Integer, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_rate_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class JobModel(Base):
    """SQLAlchemy row for Job.

    `status` is stored as the JobStatus member *name* (e.g. "RUNNING"),
    never its auto() integer value -- names are stable across enum
    reorderings, readable in the database, and reconstructed with
    JobStatus[name]. Only fields the domain Job actually has today are
    persisted; progress/output path/timestamps are not part of the
    domain entity yet.
    """

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id"), nullable=False
    )
    cutoff_hz: Mapped[float] = mapped_column(Float, nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
