"""Shareable pipeline storages with operation-local sessions and transactions."""

from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import TypeVar

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.persistence.models.critical_value import CriticalValueKey, CriticalValueModel
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.persistence.stores.base import IBulkStoreReader, IBulkStoreWriter
from pysatl_criterion.persistence.stores.critical_value import ICriticalValueStorage


T = TypeVar("T")


class DistributionStorage(
    IBulkStoreReader[LimitDistributionModel, DistributionKey, LimitDistributionFilter],
    IBulkStoreWriter[LimitDistributionModel],
):
    """Read/write distributions, committing improvements and cache invalidation together.

    The caller owns the database. Methods finish their transaction before returning; batch
    iterators release each page before yielding. Do not mutate inputs during a method call.
    """

    def __init__(self, database: AlchemyDatabase):
        self._database = database
        self._make_uow = partial(AlchemyUnitOfWork, database.session_factory)
        self._service = DistributionService(self._make_uow)

    def get(self, query: DistributionKey) -> LimitDistributionModel | None:
        def read() -> LimitDistributionModel | None:
            with self._make_uow() as uow:
                return uow.limit_distributions.get(query)

        return self._database.run(read)

    def get_batch(
        self,
        query: LimitDistributionFilter,
        *,
        batch_size: int = 100,
        after: DistributionKey | None = None,
    ) -> list[LimitDistributionModel]:
        def read() -> list[LimitDistributionModel]:
            with self._make_uow() as uow:
                return uow.limit_distributions.get_batch(query, batch_size=batch_size, after=after)

        return self._database.run(read)

    def iter_batches(
        self, query: LimitDistributionFilter, *, batch_size: int = 100
    ) -> Iterator[list[LimitDistributionModel]]:
        after: DistributionKey | None = None
        while True:
            batch = self.get_batch(query, batch_size=batch_size, after=after)
            if not batch:
                return
            after = batch[-1].key
            last = len(batch) < batch_size
            yield batch
            if last:
                return

    def insert(self, model: LimitDistributionModel) -> bool:
        return self._database.run(lambda: self._service.save_distribution(model), write=True)

    def bulk_insert(self, models: Sequence[LimitDistributionModel]) -> list[LimitDistributionModel]:
        return self._database.run(lambda: self._service.save_distributions(models), write=True)


class CriticalValueStorage(
    ICriticalValueStorage,
    IBulkStoreReader[CriticalValueModel, CriticalValueLookupQuery, CriticalValueFilter],
):
    """Read/write current cached bounds through fresh, short-lived units of work.

    Instances share the database's writer and pool with DistributionStorage. Writes commit
    before returning and never accept bounds computed from an obsolete source version.
    """

    def __init__(self, database: AlchemyDatabase):
        self._database = database
        self._make_uow = partial(AlchemyUnitOfWork, database.session_factory)

    def get(self, query: CriticalValueLookupQuery) -> CriticalValueModel | None:
        def read() -> CriticalValueModel | None:
            with self._make_uow() as uow:
                return uow.critical_values.get(query)

        return self._database.run(read)

    def get_batch(
        self,
        query: CriticalValueFilter,
        *,
        batch_size: int = 100,
        after: CriticalValueKey | None = None,
    ) -> list[CriticalValueModel]:
        def read() -> list[CriticalValueModel]:
            with self._make_uow() as uow:
                return uow.critical_values.get_batch(query, batch_size=batch_size, after=after)

        return self._database.run(read)

    def iter_batches(
        self, query: CriticalValueFilter, *, batch_size: int = 100
    ) -> Iterator[list[CriticalValueModel]]:
        after: CriticalValueKey | None = None
        while True:
            batch = self.get_batch(query, batch_size=batch_size, after=after)
            if not batch:
                return
            after = batch[-1].key
            last = len(batch) < batch_size
            yield batch
            if last:
                return

    def _write(self, operation: Callable[[ICriticalValueStorage], T]) -> T:
        def write() -> T:
            with self._make_uow() as uow:
                result = operation(uow.critical_values)
                uow.commit()
                return result

        return self._database.run(write, write=True)

    def insert(self, model: CriticalValueModel) -> bool:
        return self._write(lambda storage: storage.insert(model))

    def bulk_insert(self, models: Sequence[CriticalValueModel]) -> list[CriticalValueModel]:
        return self._write(lambda storage: storage.bulk_insert(models))

    def save_if_current(self, data: CriticalValueModel) -> bool:
        return self._write(lambda storage: storage.save_if_current(data))

    def delete_for_distribution(self, key: DistributionKey) -> None:
        self._write(lambda storage: storage.delete_for_distribution(key))
