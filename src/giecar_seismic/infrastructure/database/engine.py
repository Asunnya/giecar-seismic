from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from giecar_seismic.infrastructure.database.models import Base


def create_sqlite_engine(database_path: str | Path) -> Engine:
    """Engine for a SQLite file with FOREIGN KEY enforcement turned on.

    SQLite ignores FOREIGN KEY constraints unless `PRAGMA foreign_keys=ON`
    is issued on *every* connection -- declaring the FK in the schema
    alone does nothing. The connect hook below does that for each new
    DBAPI connection this engine hands out, so the jobs.dataset_id ->
    datasets.id relationship is actually enforced (see the repository
    tests, which prove it).

    The path is a plain parameter for now; the desktop app's default
    location becomes configurable in a later slice.
    """
    engine = create_engine(f"sqlite:///{Path(database_path)}")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: repositories convert ORM rows to plain
    # domain objects *after* commit, inside the same short-lived session.
    # Without this, committing would expire every loaded attribute and
    # the conversion would trigger a reload (or fail once the session is
    # closed). Sessions are never shared or kept open beyond one
    # repository call.
    return sessionmaker(bind=engine, expire_on_commit=False)
