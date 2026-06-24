import numpy as np
import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.hypothesis_testing.critical_values.calculator.model import (
    CriticalValueCalculator,
)


class LeftCriticalValueCalculator(CriticalValueCalculator[float]):
    @override
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> float:
        ecdf = scipy_stats.ecdf(limit_distribution)
        return float(np.quantile(ecdf.cdf.quantiles, q=significance_level))


class RightCriticalValueCalculator(CriticalValueCalculator[float]):
    @override
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> float:
        ecdf = scipy_stats.ecdf(limit_distribution)
        return float(np.quantile(ecdf.cdf.quantiles, q=1 - significance_level))


class TwoSidedCriticalValueCalculator(CriticalValueCalculator[tuple[float, float]]):
    @override
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> tuple[float, float]:
        ecdf = scipy_stats.ecdf(limit_distribution)
        sl = significance_level / 2
        left = float(np.quantile(ecdf.cdf.quantiles, q=sl))
        right = float(np.quantile(ecdf.cdf.quantiles, q=1 - sl))

        return left, right
