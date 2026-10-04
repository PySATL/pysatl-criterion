from abc import abstractmethod

from pysatl_criterion.persistence.models.critical_value import CriticalValueKey, CriticalValueModel
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.queries.critical_value import (
    CriticalValueFilter,
    CriticalValueLookupQuery,
)
from pysatl_criterion.persistence.stores.base import IBulkStoreWriter, IStoreReader


class ICriticalValueStorage(
    IStoreReader[CriticalValueModel, CriticalValueLookupQuery], IBulkStoreWriter[CriticalValueModel]
):
    """Write cached bounds only for matching local distributions and source versions.

    insert and bulk_insert report actual changes; existing equal-version bounds are retained.
    The caller owns the transaction for each operation.
    """

    @abstractmethod
    def get(self, query: CriticalValueLookupQuery) -> CriticalValueModel | None:
        """Return cached bounds only if the requested source version is still current."""

    @abstractmethod
    def get_batch(
        self,
        query: CriticalValueFilter,
        *,
        batch_size: int = 100,
        after: CriticalValueKey | None = None,
    ) -> list[CriticalValueModel]:
        """Read current bounds in primary-key order, strictly after the cursor.

        batch_size must be positive. Stale cached versions are excluded.
        """

    @abstractmethod
    def save_if_current(self, data: CriticalValueModel) -> bool:
        """Save only if the source version is current, locking it until transaction end."""

    @abstractmethod
    def delete_for_distribution(self, key: DistributionKey) -> None:
        """Invalidate every cached level and alternative for the distribution."""
