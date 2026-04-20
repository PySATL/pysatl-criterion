import logging

from pysatl_criterion.critical_value.loader.remote_loader import CriticalValueLoader
from pysatl_criterion.critical_value.resolver.model import CriticalArea, CriticalValueResolver
from pysatl_criterion.critical_value.resolver.storage_resolver import StorageCriticalValueResolver
from pysatl_criterion.statistics.models import HypothesisType


logger = logging.getLogger(__name__)


class CompositeCriticalValueResolver(CriticalValueResolver):
    """
    Critical value composite resolver.
    """

    def __init__(
        self,
        local_resolver: StorageCriticalValueResolver,
        cv_loader: CriticalValueLoader,
    ):
        self._local_resolver = local_resolver
        self._cv_loader = cv_loader
        self._storage = local_resolver.limit_distribution_storage

    def resolve_bulk(
        self,
        criterion_codes: list[str],
        sample_size: int,
        sl: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> dict[str, CriticalArea]:
        """
        Resolve multiple critical values using cache and bulk loader.
        """
        # 1. Get all local results
        results = self._local_resolver.resolve_bulk(criterion_codes, sample_size, sl, alternative)

        # 2. Search for missing results
        missing = [c for c in criterion_codes if c not in results]

        # 3. Load missing results and update the results dictionary
        if missing:
            for code in missing:
                self._cv_loader.load(code, sample_size)
            new_data = self._local_resolver.resolve_bulk(missing, sample_size, sl, alternative)
            results.update(new_data)

        return results
