"""Bounded SQLite writes and retries of complete, fresh transactions."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Any, TypeVar

from sqlalchemy.exc import DBAPIError


T = TypeVar("T")


class SQLiteWriter:
    def __init__(self, queue_size: int, timeout: float):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pysatl-sqlite")
        # One running task plus a bounded number of pending tasks.
        self._slots = BoundedSemaphore(queue_size + 1)
        self._timeout = timeout

    def run(self, operation: Callable[[], T]) -> T:
        if not self._slots.acquire(timeout=self._timeout):
            raise TimeoutError("Timed out waiting for the SQLite write queue")
        try:
            future = self._executor.submit(operation)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(self._release)
        # Return only after commit, or propagate the original transaction exception.
        return future.result()

    def _release(self, _future: Future[Any]) -> None:
        self._slots.release()

    def close(self) -> None:
        self._executor.shutdown(wait=True)


def retryable(error: DBAPIError) -> bool:
    """Recognize lock/deadlock/serialization failures, never uncertain disconnects."""
    if error.connection_invalidated:
        return False
    original = error.orig
    code = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if code in ("40001", "40P01"):
        return True
    sqlite_code = getattr(original, "sqlite_errorcode", None)
    if isinstance(sqlite_code, int) and sqlite_code & 0xFF in (5, 6):
        return True
    args = getattr(original, "args", ())
    if args and args[0] in (1205, 1213):
        return True
    # Python 3.10 sqlite3 exceptions do not expose sqlite_errorcode.
    return str(original) in ("database is locked", "database table is locked")
