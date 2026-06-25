import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.hypothesis_testing.critical_values.calculator.model import (
    CriticalValueCalculator,
)


class LeftCriticalValueCalculator(CriticalValueCalculator[float]):
    """
    Critical value calculator for left-tailed tests.
    """

    @override
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> float:
        """
        Calculate the lower-tail critical value.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param significance_level: test significance level.
        :return: critical value at the lower-tail significance quantile.
        """
        ecdf = scipy_stats.ecdf(limit_distribution)
        return float(np.quantile(ecdf.cdf.quantiles, q=significance_level))


class RightCriticalValueCalculator(CriticalValueCalculator[float]):
    """
    Critical value calculator for right-tailed tests.
    """

    @override
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> float:
        """
        Calculate the upper-tail critical value.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param significance_level: test significance level.
        :return: critical value at the upper-tail significance quantile.
        """
        ecdf = scipy_stats.ecdf(limit_distribution)
        return float(np.quantile(ecdf.cdf.quantiles, q=1 - significance_level))


class TwoSidedCriticalValueCalculator(CriticalValueCalculator[tuple[float, float]]):
    """
    Critical value calculator for two-sided tests.
    """

    @override
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> tuple[float, float]:
        """
        Calculate lower and upper critical values for a two-sided test.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param significance_level: test significance level split between both tails.
        :return: tuple of lower and upper critical values.
        """
        ecdf = scipy_stats.ecdf(limit_distribution)
        sl = significance_level / 2
        left = float(np.quantile(ecdf.cdf.quantiles, q=sl))
        right = float(np.quantile(ecdf.cdf.quantiles, q=1 - sl))

        return left, right
