from abc import ABC, abstractmethod

from typing_extensions import override

from pysatl_criterion import IStatistic


class AbstractUniformityStatistic(IStatistic, ABC):
    @staticmethod
    @override
    def code():
        return "UNIFORMITY"

    @abstractmethod
    def execute_statistic(self, x, y, **kwargs) -> float:
        """
        Execute uniformity test statistic and return calculated statistic value.

        :param x: rvs data to calculated statistic value
        :param y: rvs data to calculated statistic value
        :param kwargs: arguments for statistic calculation
        """
        raise NotImplementedError("Method is not implemented")
