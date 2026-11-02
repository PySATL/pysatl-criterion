from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pysatl_criterion.loader.models import LoadResult
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.stores.base import IBulkStoreReader, IBulkStoreWriter


M = TypeVar("M")
Q = TypeVar("Q")
F = TypeVar("F", bound=LimitDistributionFilter)


class AbstractLoader(ABC, Generic[M, Q, F]):
    """Share batch transfer and counting between concrete store loaders.

    The source needs only reading and the destination only writing. Use a transactional writer
    adapter at the destination to invalidate critical values along with distribution updates.
    Each batch is committed independently. Read/write errors propagate; earlier batches remain
    committed when a later batch fails.
    """

    def __init__(
        self,
        source: IBulkStoreReader[M, Q, F],
        destination: IBulkStoreWriter[M],
        *,
        print_summary: bool = True,
    ):
        self._source: IBulkStoreReader[M, Q, F] = source
        self._destination: IBulkStoreWriter[M] = destination
        self._print_summary = print_summary

    @staticmethod
    @abstractmethod
    def _criterion_code(model: M) -> str:
        """Extract the criterion code used to track missing source matches."""

    def load(self, query: F, *, batch_size: int = 100) -> LoadResult:
        """Transfer bounded batches, saving each in its own destination transaction."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        fetched_count = saved_count = 0
        missing_codes = dict.fromkeys(
            criterion.criterion_code for criterion in query.criteria or ()
        )
        for models in self._source.iter_batches(query, batch_size=batch_size):
            if not models:
                continue
            fetched_count += len(models)
            for model in models:
                missing_codes.pop(self._criterion_code(model), None)
            saved_count += len(self._destination.bulk_insert(models))
        result = LoadResult(fetched_count, saved_count, list(missing_codes))
        if self._print_summary:
            record_label = "record" if result.saved_count == 1 else "records"
            print(
                f"Loaded {result.saved_count} {record_label} "
                f"(fetched: {result.fetched_count}, skipped: {result.skipped_count})."
            )
        return result
