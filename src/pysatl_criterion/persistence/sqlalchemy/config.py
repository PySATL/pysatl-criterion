"""Connection and concurrency limits for a process-local database."""

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class StorageConfig:
    max_workers: int = 30
    pool_size: int | None = None
    max_overflow: int = 0
    pool_timeout: float = 30
    pool_pre_ping: bool = True
    pool_recycle: int = -1
    sqlite_timeout: float = 30
    write_queue_size: int = 30
    write_queue_timeout: float = 30
    retry_attempts: int = 3
    retry_backoff: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.pool_pre_ping, bool):
            raise TypeError("pool_pre_ping must be a boolean")
        for name in ("max_workers", "write_queue_size", "retry_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.pool_size is not None and (
            isinstance(self.pool_size, bool)
            or not isinstance(self.pool_size, int)
            or self.pool_size <= 0
        ):
            raise ValueError("pool_size must be a positive integer or None")
        if (
            isinstance(self.max_overflow, bool)
            or not isinstance(self.max_overflow, int)
            or self.max_overflow < 0
        ):
            raise ValueError("max_overflow must be a nonnegative integer")
        if (
            isinstance(self.pool_recycle, bool)
            or not isinstance(self.pool_recycle, int)
            or self.pool_recycle < -1
        ):
            raise ValueError("pool_recycle must be -1 or a nonnegative integer")
        for name in ("pool_timeout", "sqlite_timeout", "write_queue_timeout", "retry_backoff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite positive number")
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")

    def engine_options(self, *, sqlite: bool, memory: bool) -> dict[str, Any]:
        options: dict[str, Any] = {
            "pool_pre_ping": self.pool_pre_ping,
            "pool_recycle": self.pool_recycle,
        }
        if not memory:
            options.update(
                pool_size=self.pool_size or (5 if sqlite else self.max_workers),
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
            )
        if sqlite:
            options["connect_args"] = {"timeout": self.sqlite_timeout}
        return options
