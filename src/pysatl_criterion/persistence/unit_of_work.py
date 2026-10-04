from abc import ABC, abstractmethod
from types import TracebackType

from pysatl_criterion.persistence.stores.critical_value import ICriticalValueStorage
from pysatl_criterion.persistence.stores.limit_distribution import ILimitDistributionStorage


class IUnitOfWork(ABC):
    limit_distributions: ILimitDistributionStorage
    critical_values: ICriticalValueStorage

    @abstractmethod
    def __enter__(self) -> "IUnitOfWork":
        """Open a transaction shared by both stores."""

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Roll back uncommitted work and release resources."""

    @abstractmethod
    def commit(self) -> None:
        """Commit both stores atomically."""

    @abstractmethod
    def rollback(self) -> None:
        """Roll back both stores."""
