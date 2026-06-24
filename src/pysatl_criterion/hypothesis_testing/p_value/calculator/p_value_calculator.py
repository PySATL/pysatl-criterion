import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator


class LeftPValueCalculator(PValueCalculator):
    @override
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:

        ecdf = scipy_stats.ecdf(limit_distribution)
        cdf_value = float(ecdf.cdf.evaluate(statistic_value))
        return cdf_value


class RightPValueCalculator(PValueCalculator):
    @override
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:

        ecdf = scipy_stats.ecdf(limit_distribution)
        cdf_value = float(ecdf.cdf.evaluate(statistic_value))
        return 1.0 - cdf_value


class TwoSidedPValueCalculator(PValueCalculator):
    @override
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:

        ecdf = scipy_stats.ecdf(limit_distribution)
        cdf_value = float(ecdf.cdf.evaluate(statistic_value))
        return 2.0 * min(cdf_value, 1.0 - cdf_value)
