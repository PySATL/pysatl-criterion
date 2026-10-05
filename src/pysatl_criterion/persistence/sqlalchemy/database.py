from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from pysatl_criterion.persistence.sqlalchemy.base import Base


class AlchemyDatabase:
    """Connection and session ownership using the configured SQLAlchemy dialect."""

    def __init__(self, connection_string: str, **engine_options: Any):
        self.engine = create_engine(connection_string, **engine_options)
        if self.engine.dialect.name == "sqlite":

            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _record):
                connection.isolation_level = None
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            @event.listens_for(self.engine, "begin")
            def begin_sqlite(connection):
                # SQLite has no row locks: reserve the writer before reading a version.
                # Calculation is always performed outside this short transaction.
                connection.exec_driver_sql("BEGIN IMMEDIATE")

        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init(self) -> None:
        # Register all tables before create_all, regardless of store import order.
        from pysatl_criterion.persistence.sqlalchemy.models import critical_value  # noqa: F401
        from pysatl_criterion.persistence.sqlalchemy.models import limit_distribution  # noqa: F401

        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()
