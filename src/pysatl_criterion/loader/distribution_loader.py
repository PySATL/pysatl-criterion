from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

from pysatl_criterion.hypothesis_testing.distribution_service import DistributionService
from pysatl_criterion.loader.critical_value_loader import CriticalValueLoader
from pysatl_criterion.loader.limit_distribution_loader import LimitDistributionLoader
from pysatl_criterion.loader.models import DistributionLoadResult
from pysatl_criterion.loader.store_adapters import (
    CriticalValueReader,
    CriticalValueWriter,
    LimitDistributionReader,
    LimitDistributionWriter,
)
from pysatl_criterion.persistence.queries.critical_value import CriticalValueFilter
from pysatl_criterion.persistence.queries.limit_distribution import LimitDistributionFilter
from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
from pysatl_criterion.persistence.sqlalchemy.unit_of_work import AlchemyUnitOfWork
from pysatl_criterion.utils import constants


@contextmanager
def _open_database(connection: str | AlchemyDatabase) -> Iterator[AlchemyDatabase]:
    if isinstance(connection, AlchemyDatabase):
        yield connection
        return
    database = AlchemyDatabase(connection)
    try:
        yield database
    finally:
        database.dispose()


class DistributionLoader:
    """Load all selected distributions before transferring their cached critical values.

    remote and local accept database URLs or existing AlchemyDatabase objects. An omitted remote
    uses constants.REMOTE_PYSATL_URL; local is required. Connections
    created from URLs are owned by each load call; passed database objects remain caller-owned.
    Each batch commits independently. A distribution-stage error prevents the critical-value
    stage from starting; any error propagates while earlier committed batches remain saved.
    """

    def __init__(
        self,
        remote: str | AlchemyDatabase | None = None,
        local: str | AlchemyDatabase | None = None,
    ):
        if local is None:
            raise TypeError("local is required")
        self._remote = constants.REMOTE_PYSATL_URL if remote is None else remote
        self._local = local

    def load(
        self, query: LimitDistributionFilter | None = None, *, batch_size: int = 100
    ) -> DistributionLoadResult:
        """Apply the same distribution filter to both stages and print one final summary.

        Without a query, transfer all distributions and their cached values.
        A CriticalValueFilter additionally restricts cached levels and alternatives.
        A plain LimitDistributionFilter transfers all matching cached levels and alternatives.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if query is None:
            query = LimitDistributionFilter()
        critical_value_query = (
            query
            if isinstance(query, CriticalValueFilter)
            else CriticalValueFilter(
                criteria=query.criteria,
                min_sample_size=query.min_sample_size,
                max_sample_size=query.max_sample_size,
                min_monte_carlo_count=query.min_monte_carlo_count,
            )
        )
        with _open_database(self._remote) as remote, _open_database(self._local) as local:
            local.init()
            remote_uow = partial(AlchemyUnitOfWork, remote.session_factory)
            local_uow = partial(AlchemyUnitOfWork, local.session_factory)
            limit_distribution_loader = LimitDistributionLoader(
                source=LimitDistributionReader(remote_uow),
                destination=LimitDistributionWriter(DistributionService(local_uow)),
                print_summary=False,
            )
            critical_value_loader = CriticalValueLoader(
                source=CriticalValueReader(remote_uow),
                destination=CriticalValueWriter(local_uow),
                print_summary=False,
            )
            distributions = limit_distribution_loader.load(query, batch_size=batch_size)
            critical_values = critical_value_loader.load(
                critical_value_query, batch_size=batch_size
            )
        result = DistributionLoadResult(distributions, critical_values)
        record_label = "record" if result.saved_count == 1 else "records"
        print(
            f"Loaded {result.saved_count} {record_label} "
            f"(limit distributions: {distributions.saved_count}, "
            f"critical values: {critical_values.saved_count}; "
            f"fetched: {result.fetched_count}, skipped: {result.skipped_count})."
        )
        return result
