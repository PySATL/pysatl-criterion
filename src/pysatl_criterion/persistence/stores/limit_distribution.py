from abc import abstractmethod

from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.stores.base import IBulkStoreWriter, IStoreReader


class ILimitDistributionStorage(
    IStoreReader[LimitDistributionModel, DistributionKey],
    IBulkStoreWriter[LimitDistributionModel],
):
    """Read distributions by key/filter and save only strictly larger simulation counts.

    Batch writes keep the largest incoming count for each key, retaining the first model on
    ties, and return only accepted models. The caller owns the transaction and must invalidate
    critical values for changed distributions in the same unit of work.
    """

    @abstractmethod
    def get_batch(
        self,
        query: LimitDistributionFilter,
        *,
        batch_size: int = 100,
        after: DistributionKey | None = None,
    ) -> list[LimitDistributionModel]:
        """Read at most batch_size matches in primary-key order, strictly after a key.

        batch_size must be positive. The caller owns the transaction; the final
        model's key can be used as the cursor for the next batch.
        """
