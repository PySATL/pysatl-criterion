import logging
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override

from pysatl_criterion.persistence.models.limit_distribution import (
    ILimitDistributionStorage,
    LimitDistributionQuery,
)
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.utils.generator import get_available_generator


logger = logging.getLogger(__name__)


class AbstractLimitDistributionResolver(ABC):
    """
    Abstract resolver for goodness-of-fit statistic limit distributions.
    """

    @abstractmethod
    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float] | None:
        """
        Resolve limit distribution values for a statistic and sample size.

        :param statistic: goodness-of-fit statistic definition.
        :param sample_size: sample size used for the limit distribution.
        :return: resolved limit distribution statistic values.
        """
        raise NotImplementedError("Method not implemented")


class MonteCarloLimitDistributionResolver(AbstractLimitDistributionResolver):
    """
    Resolver that estimates limit distributions with Monte Carlo simulation.
    """

    def __init__(
        self,
        monte_carlo_count: int,
    ):
        """
        Initialize the Monte Carlo resolver.

        :param monte_carlo_count: number of simulated samples to generate.
        :raises ValueError: if monte_carlo_count is not positive.
        """
        if monte_carlo_count <= 0:
            raise ValueError("monte_carlo_count must be positive")

        self.monte_carlo_count = monte_carlo_count

    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float]:
        """
        Estimate a limit distribution for the statistic by simulation.

        :param statistic: goodness-of-fit statistic definition.
        :param sample_size: simulated sample size.
        :return: simulated statistic values.
        :raises ValueError: if sample_size is not positive.
        """
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

        statistics = np.empty(self.monte_carlo_count)

        rvs_generator = get_available_generator(
            statistic.distribution(), statistic.hypothesis().params
        )

        for i in range(self.monte_carlo_count):
            statistics[i] = statistic.execute_statistic(rvs_generator.generate(sample_size))

        return list(statistics)


class StorageLimitDistributionResolver(AbstractLimitDistributionResolver):
    """
    Resolver that loads limit distributions from persistent storage.
    """

    def __init__(self, limit_distribution_storage: ILimitDistributionStorage):
        """
        Initialize the storage-backed resolver.

        :param limit_distribution_storage: limit distribution storage.
        """
        self.limit_distribution_storage = limit_distribution_storage

    @override
    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float] | None:
        """
        Resolve limit distribution values from storage.

        :param statistic: goodness-of-fit statistic definition.
        :param sample_size: sample size used for the stored distribution.
        :return: stored statistic values, or None if no distribution is found.
        """
        limit_distribution = self.limit_distribution_storage.get_data(
            LimitDistributionQuery(
                criterion_code=statistic.code(),
                criterion_parameters=statistic.hypothesis().parameters(),
                sample_size=sample_size,
                monte_carlo_count=1,
            )
        )

        return (
            list(limit_distribution.results_statistics) if limit_distribution is not None else None
        )


class CompositeLimitDistributionResolver(AbstractLimitDistributionResolver):
    def __init__(
        self,
        local_resolver: StorageLimitDistributionResolver,
        monte_carlo_resolver: MonteCarloLimitDistributionResolver,
    ):
        self._local_resolver = local_resolver
        self._monte_carlo_resolver = monte_carlo_resolver

    def resolve(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        sample_size: int,
    ) -> list[float]:
        # 1. Get all local results
        results = self._local_resolver.resolve(statistic, sample_size)

        if results is not None:
            return results

        # 2. Monte-Carlo limit distribution calculation
        results = self._monte_carlo_resolver.resolve(statistic, sample_size)

        return results
