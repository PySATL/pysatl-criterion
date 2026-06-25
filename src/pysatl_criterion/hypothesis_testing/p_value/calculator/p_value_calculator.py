import scipy.stats as scipy_stats
from typing_extensions import override

from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator


class LeftPValueCalculator(PValueCalculator):
    """
    P-value calculator for left-tailed tests.
    """

    @override
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:
        """
        Calculate the lower-tail p-value for a statistic value.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param statistic_value: computed statistic value for the observed sample.
        :return: lower-tail p-value.
        """
        ecdf = scipy_stats.ecdf(limit_distribution)
        cdf_value = float(ecdf.cdf.evaluate(statistic_value))
        return cdf_value


class RightPValueCalculator(PValueCalculator):
    """
    P-value calculator for right-tailed tests.
    """

    @override
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:
        """
        Calculate the upper-tail p-value for a statistic value.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param statistic_value: computed statistic value for the observed sample.
        :return: upper-tail p-value.
        """
        ecdf = scipy_stats.ecdf(limit_distribution)
        cdf_value = float(ecdf.cdf.evaluate(statistic_value))
        return 1.0 - cdf_value


class TwoSidedPValueCalculator(PValueCalculator):
    """
    P-value calculator for two-sided tests.
    """

    @override
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:
        """
        Calculate the two-sided p-value for a statistic value.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param statistic_value: computed statistic value for the observed sample.
        :return: two-sided p-value.
        """
        ecdf = scipy_stats.ecdf(limit_distribution)
        cdf_value = float(ecdf.cdf.evaluate(statistic_value))
        return 2.0 * min(cdf_value, 1.0 - cdf_value)
