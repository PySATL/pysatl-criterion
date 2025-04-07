from abc import ABC, abstractmethod

from typing_extensions import override

from pysatl_criterion.model import IStatistic


class AbstractGoodnessOfFitStatistic(IStatistic, ABC):
    @staticmethod
    @override
    def code():
        return "GOODNESS_OF_FIT"

    @abstractmethod
    def execute_statistic(self, x, **kwargs: dict[str, any] | None) -> float:
        """
        Execute test statistic and return calculated statistic value.

        :param x: rvs data to calculated statistic value
        :param kwargs: arguments for statistic calculation
        """
        raise NotImplementedError("Method is not implemented")
