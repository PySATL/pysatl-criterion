import os
from collections.abc import Callable
from threading import Condition, get_ident, local
from time import sleep
from typing import Any, TypeVar

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from pysatl_criterion.persistence.sqlalchemy._execution import SQLiteWriter, retryable
from pysatl_criterion.persistence.sqlalchemy.base import Base
from pysatl_criterion.persistence.sqlalchemy.config import StorageConfig


T = TypeVar("T")


class AlchemyDatabase:
    """Connection and session ownership using the configured SQLAlchemy dialect."""

    def __init__(
        self, connection_string: str, *, config: StorageConfig | None = None, **engine_options: Any
    ):
        self.config = config or StorageConfig()
        self._pid = os.getpid()
        self._owner_thread = get_ident()
        self._condition = Condition()
        self._execution = local()
        self._active = 0
        self._closing = False
        self._closed = False
        url = make_url(connection_string)
        sqlite = url.get_backend_name() == "sqlite"
        self._memory = sqlite and (
            url.database in (None, "", ":memory:")
            or (url.database or "").startswith("file::memory:")
            or url.query.get("mode") == "memory"
        )
        options = self.config.engine_options(sqlite=sqlite, memory=self._memory)
        if "poolclass" in engine_options or "pool" in engine_options:
            for name in ("pool_size", "max_overflow", "pool_timeout"):
                options.pop(name, None)
        if "pool" in engine_options:
            for name in ("pool_pre_ping", "pool_recycle"):
                options.pop(name, None)
        connect_args = {**options.get("connect_args", {}), **engine_options.get("connect_args", {})}
        options.update(engine_options)
        if sqlite or connect_args:
            options["connect_args"] = connect_args
        self.engine = create_engine(connection_string, **options)
        self._writer = (
            SQLiteWriter(self.config.write_queue_size, self.config.write_queue_timeout)
            if sqlite and not self._memory
            else None
        )
        if self.engine.dialect.name == "sqlite":

            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _record):
                connection.isolation_level = None
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            @event.listens_for(self.engine, "begin")
            def begin_sqlite(connection):
                # Readers must not reserve the single writer slot. Both guarded write
                # paths start with UPDATE, acquiring the write lock before version reads.
                connection.exec_driver_sql("BEGIN")

        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init(self) -> None:
        self.run(self._initialize)

    def _initialize(self) -> None:
        # Register all tables before create_all, regardless of store import order.
        from pysatl_criterion.persistence.sqlalchemy.models import (
            critical_value,  # noqa: F401
            limit_distribution,  # noqa: F401
        )
        from pysatl_criterion.persistence.sqlalchemy.schema import require_configuration_schema

        require_configuration_schema(self.engine)
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self._check_process()
        if getattr(self._execution, "active", False):
            raise RuntimeError("Cannot dispose a database inside its operation")
        with self._condition:
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._active == 0)
        try:
            if self._writer is not None:
                self._writer.close()
            self.engine.dispose()
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def run(self, operation: Callable[[], T], *, write: bool = False) -> T:
        """Run a short operation whose callable opens a fresh UoW on every attempt.

        Public storages use this boundary; direct session/UoW users retain transaction ownership.
        """
        self._check_process()
        if getattr(self._execution, "active", False):
            raise RuntimeError("Cannot nest public storage/database operations")
        with self._condition:
            if self._closing:
                raise RuntimeError("Database is closed or shutting down")
            if self._memory and get_ident() != self._owner_thread:
                raise RuntimeError("Use a file-backed SQLite database for threaded storages")
            self._active += 1
        try:
            if write and self._writer is not None:
                return self._writer.run(lambda: self._retry(operation))
            return self._retry(operation)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _retry(self, operation: Callable[[], T]) -> T:
        self._execution.active = True
        try:
            return self._attempt(operation)
        finally:
            self._execution.active = False

    def _attempt(self, operation: Callable[[], T]) -> T:
        for attempt in range(self.config.retry_attempts):
            try:
                return operation()
            except DBAPIError as error:
                if not retryable(error) or attempt + 1 == self.config.retry_attempts:
                    raise
                sleep(self.config.retry_backoff * (2**attempt))
        raise AssertionError("Unreachable retry state")

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("Create AlchemyDatabase inside each worker process")
