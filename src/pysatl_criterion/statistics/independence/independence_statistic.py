"""Base class for independence statistics."""

from abc import ABC, abstractmethod

from numpy import float64
from typing_extensions import override

from pysatl_criterion.statistics.hypothesis import IndependenceHypothesis
from pysatl_criterion.statistics.statistic import AbstractStatistic


class AbstractIndependenceStatistic(AbstractStatistic, ABC):
    """
    Abstract base class for independence statistics.
    """

    @abstractmethod
    def hypothesis(self) -> IndependenceHypothesis:
        """
        Get hypothesis.

        :return: hypothesis.
        """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for independence statistics.

        :return: string code "INDEPENDENCE".
        """
        return "INDEPENDENCE"

    @abstractmethod
    def execute_statistic(self, rvs1, rvs2) -> float | float64:
        """
        Execute test statistic and return calculated statistic value.

        :param rvs1: rvs data to calculated statistic value
        :param rvs2: rvs data to calculated statistic value
        """
        raise NotImplementedError("Method is not implemented")
