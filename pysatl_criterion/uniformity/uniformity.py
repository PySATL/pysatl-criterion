from numpy import float64
from scipy.stats import ks_2samp
from typing_extensions import override

from pysatl_criterion.uniformity.model import AbstractUniformityStatistic


class KolmogorovSmirnovUniformityStatistic(AbstractUniformityStatistic):
    @override
    def execute_statistic(self, x, y, **kwargs) -> float | float64:
        return ks_2samp(x, y).statistic

    @staticmethod
    @override
    def code():
        return f"KS_{AbstractUniformityStatistic.code()}"


class SmirnovLessUniformityStatistic(AbstractUniformityStatistic):
    @override
    def execute_statistic(self, x, y, **kwargs) -> float | float64:
        return ks_2samp(x, y).statistic

    @staticmethod
    @override
    def code():
        return f"SMIRNOV_LESS_{AbstractUniformityStatistic.code()}"


class SmirnovGreaterUniformityStatistic(AbstractUniformityStatistic):
    @override
    def execute_statistic(self, x, y, **kwargs) -> float | float64:
        return ks_2samp(x, y).statistic

    @staticmethod
    @override
    def code():
        return f"SMIRNOV_GRATER_{AbstractUniformityStatistic.code()}"
