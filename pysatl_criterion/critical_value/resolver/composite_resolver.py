from typing_extensions import override

from pysatl_criterion.critical_value.loader.remote_loader import CriticalValueLoader
from pysatl_criterion.critical_value.resolver.model import CriticalArea, CriticalValueResolver
from pysatl_criterion.critical_value.resolver.storage_resolver import StorageCriticalValueResolver
from pysatl_criterion.statistics.models import HypothesisType


class CompositeCriticalValueResolver(CriticalValueResolver):
    """
    Critical value composite resolver.
    """

    def __init__(
        self,
        local_resolver: StorageCriticalValueResolver,
        remote_resolver: StorageCriticalValueResolver,
    ):
        self._local_resolver = local_resolver
        self._remote_resolver = remote_resolver

    @override
    def resolve(
        self,
        criterion_code: str,
        sample_size: int,
        sl: float,
        alternative: HypothesisType = HypothesisType.RIGHT,
    ) -> CriticalArea | None:
        """
        Resolve critical value for given criterion.
            1. Try to get local value
            2. Try to get remote value and cache it to local storage.

        :param criterion_code: criterion code.
        :param sample_size: sample size.
        :param sl: significance level.
        :param alternative: test alternative

        :return: critical value.
        """

        # 1. Try to get local value
        result = self._local_resolver.resolve(criterion_code, sample_size, sl, alternative)

        if result is not None:
            return result

        # 2. Try to get remote value and cache it to local storage.
        CriticalValueLoader(
            self._local_resolver.limit_distribution_storage,
            self._remote_resolver.limit_distribution_storage,
        ).load(criterion_code, sample_size)

        return self._local_resolver.resolve(criterion_code, sample_size, sl, alternative)
