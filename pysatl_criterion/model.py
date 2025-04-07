from abc import ABC, abstractmethod

from numpy import float64


class IStatistic(ABC):
    @staticmethod
    @abstractmethod
    def code() -> str:
        """
        Generate unique code for test statistic.
        """
        raise NotImplementedError("Method is not implemented")

    @abstractmethod
    def execute_statistic(self, *args, **kwargs) -> float | float64:
        """
        Execute test statistic and return calculated statistic value.

        :param args: rvs data to calculated statistic value
        :param kwargs: arguments for statistic calculation
        """
        raise NotImplementedError("Method is not implemented")
