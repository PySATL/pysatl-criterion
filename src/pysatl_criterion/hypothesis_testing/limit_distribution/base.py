from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override

from pysatl_criterion.persistence.models.limit_distribution import (
    ILimitDistributionStorage,
    LimitDistributionQuery,
)
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.utils.generator import get_available_generator


class AbstractLimitDistributionResolver(ABC):
    @abstractmethod
    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float]:
        raise NotImplementedError("Method not implemented")


class MonteCarloLimitDistributionResolver(AbstractLimitDistributionResolver):
    def __init__(
        self,
        monte_carlo_count: int,
    ):
        if monte_carlo_count <= 0:
            raise ValueError("monte_carlo_count must be positive")

        self.monte_carlo_count = monte_carlo_count

    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float]:

        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

        statistics = np.empty(self.monte_carlo_count)

        rvs_generator = get_available_generator(
            statistic.distribution(), statistic.hypothesis().parameters
        )

        for i in range(self.monte_carlo_count):
            statistics[i] = statistic.execute_statistic(rvs_generator.generate(sample_size))

        return list(statistics)


class StorageLimitDistributionResolver(AbstractLimitDistributionResolver):
    """
    Critical value resolver.

    :param limit_distribution_storage: limit distribution storage
    """

    def __init__(self, limit_distribution_storage: ILimitDistributionStorage):
        self.limit_distribution_storage = limit_distribution_storage

    @override
    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float]:
        limit_distribution = self.limit_distribution_storage.get_data(
            LimitDistributionQuery(
                criterion_code=statistic.code(),
                criterion_parameters=statistic.hypothesis().parameters(),
                sample_size=sample_size,
                monte_carlo_count=1,
            )
        )

        return list(limit_distribution.result_statistic) if limit_distribution is not None else None

""""
class CompositeLimitDistributionResolver(AbstractLimitDistributionResolver):

    def __init__(
        self,
        local_resolver: StorageLimitDistributionResolver,
        cv_loader: CriticalValueLoader,
    ):
        self._local_resolver = local_resolver
        self._cv_loader = cv_loader
        self._storage = local_resolver.limit_distribution_storage

    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float]:
        # 1. Get all local results
        results = self._local_resolver.resolve_bulk(criterion_codes, sample_size, sl, alternative)

        # 2. Search for missing results
        missing = [c for c in criterion_codes if c not in results]

        # 3. Load missing results and update the results dictionary
        if missing:
            load_result = self._cv_loader.load_bulk(missing, sample_size)

            if load_result.newly_cached_count > 0:
                logger.info(
                    f"Loaded {load_result.newly_cached_count} new criteria. "
                    f"Retrying local resolution..."
                )
                codes_to_retry = [c for c in missing if c not in load_result.not_found_codes]
                if codes_to_retry:
                    new_results = self._local_resolver.resolve_bulk(
                        codes_to_retry, sample_size, sl, alternative
                    )
                    results.update(new_results)
            else:
                logger.warning(
                    f"Could not load any new data. Missing: {load_result.not_found_codes}"
                )

        return results
"""