from collections.abc import Callable, Iterator, Sequence

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.persistence.models.critical_value import CriticalValueKey, CriticalValueModel
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.stores.base import IBulkStoreReader, IBulkStoreWriter
from pysatl_criterion.persistence.unit_of_work import IUnitOfWork


class LimitDistributionReader(
    IBulkStoreReader[LimitDistributionModel, DistributionKey, LimitDistributionFilter]
):
    """Open a short unit of work for each read from an application data source."""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork]):
        self._uow_factory = uow_factory

    def get(self, query: DistributionKey) -> LimitDistributionModel | None:
        with self._uow_factory() as uow:
            return uow.limit_distributions.get(query)

    def iter_batches(
        self, query: LimitDistributionFilter, *, batch_size: int = 100
    ) -> Iterator[list[LimitDistributionModel]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        after: DistributionKey | None = None
        while True:
            with self._uow_factory() as uow:
                models = uow.limit_distributions.get_batch(
                    query, batch_size=batch_size, after=after
                )
            if not models:
                return
            after = models[-1].key
            last_batch = len(models) < batch_size
            yield models
            if last_batch:
                return


class LimitDistributionWriter(IBulkStoreWriter[LimitDistributionModel]):
    """Commit writes and critical-value invalidation together through the service."""

    def __init__(self, service: DistributionService):
        self._service = service

    def insert(self, model: LimitDistributionModel) -> bool:
        return self._service.save_distribution(model)

    def bulk_insert(self, models: Sequence[LimitDistributionModel]) -> list[LimitDistributionModel]:
        return self._service.save_distributions(models)


class CriticalValueReader(
    IBulkStoreReader[CriticalValueModel, CriticalValueLookupQuery, CriticalValueFilter]
):
    """Read current cached bounds, closing the source transaction before each batch."""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork]):
        self._uow_factory = uow_factory

    def get(self, query: CriticalValueLookupQuery) -> CriticalValueModel | None:
        with self._uow_factory() as uow:
            return uow.critical_values.get(query)

    def iter_batches(
        self, query: CriticalValueFilter, *, batch_size: int = 100
    ) -> Iterator[list[CriticalValueModel]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        after: CriticalValueKey | None = None
        while True:
            with self._uow_factory() as uow:
                models = uow.critical_values.get_batch(query, batch_size=batch_size, after=after)
            if not models:
                return
            after = models[-1].key
            last_batch = len(models) < batch_size
            yield models
            if last_batch:
                return


class CriticalValueWriter(IBulkStoreWriter[CriticalValueModel]):
    """Commit each batch of bounds whose versions match local distributions."""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork]):
        self._uow_factory = uow_factory

    def insert(self, model: CriticalValueModel) -> bool:
        with self._uow_factory() as uow:
            changed = uow.critical_values.insert(model)
            uow.commit()
            return changed

    def bulk_insert(self, models: Sequence[CriticalValueModel]) -> list[CriticalValueModel]:
        with self._uow_factory() as uow:
            saved = uow.critical_values.bulk_insert(models)
            uow.commit()
            return saved
