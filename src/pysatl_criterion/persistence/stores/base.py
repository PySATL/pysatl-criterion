from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar


M = TypeVar("M")
Q_contra = TypeVar("Q_contra", contravariant=True)
BQ_contra = TypeVar("BQ_contra", contravariant=True)


class IStoreReader(ABC, Generic[M, Q_contra]):
    @abstractmethod
    def get(self, query: Q_contra) -> M | None:
        """Read one model by its identity."""


class IBulkStoreReader(IStoreReader[M, Q_contra], Generic[M, Q_contra, BQ_contra]):
    @abstractmethod
    def iter_batches(self, query: BQ_contra, *, batch_size: int = 100) -> Iterator[list[M]]:
        """Yield nonempty batches, releasing source transactions before each yield.

        batch_size must be positive. Each batch contains at most batch_size models;
        the iteration does not promise a single snapshot across batches.
        """


class IStoreWriter(ABC, Generic[M]):
    @abstractmethod
    def insert(self, model: M) -> bool:
        """Save a model according to the store's policy; return whether it changed."""


class IBulkStoreWriter(IStoreWriter[M]):
    @abstractmethod
    def bulk_insert(self, models: Sequence[M]) -> list[M]:
        """Save a batch; return the accepted model for each identity that changed."""
