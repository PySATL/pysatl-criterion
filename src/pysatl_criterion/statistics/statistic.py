from abc import ABC, abstractmethod

from numpy import float64
from typing_extensions import override

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics.alternative import Alternative
from pysatl_criterion.statistics.hypothesis import (
    GoodnessOfFitHypothesis,
    Hypothesis,
    IndependenceHypothesis,
)


class AbstractStatistic(ABC):
    @abstractmethod
    def hypothesis(self) -> Hypothesis:
        """
        Get alternative type.

        :return: alternative type.
        """
        pass

    @abstractmethod
    def alternative(self) -> Alternative:
        """
        Get alternative.

        :return: alternative.
        """
        pass

    @staticmethod
    @abstractmethod
    def code() -> str:
        """
        Generate unique code for test statistic.
        """
        raise NotImplementedError("Method is not implemented")

    @staticmethod
    @abstractmethod
    def short_code():
        """
        Generate non-unique short code for test statistic.
        """
        raise NotImplementedError("Method is not implemented")


class AbstractGoodnessOfFitStatistic(AbstractStatistic, ABC):
    """
    Abstract base class for goodness-of-fit statistics.
    """

    @abstractmethod
    def hypothesis(self) -> GoodnessOfFitHypothesis:
        """
        Get hypothesis.

        :return: hypothesis.
        """
        pass

    @staticmethod
    @abstractmethod
    def distribution() -> DistributionType:
        """
        Get distribution type.

        :return: DistributionType.
        """
        pass

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for goodness-of-fit statistics.

        :return: string code "GOODNESS_OF_FIT".
        """
        return "GOODNESS_OF_FIT"

    @abstractmethod
    def execute_statistic(self, rvs) -> float | float64:
        """
        Execute test statistic and return calculated statistic value.

        :param rvs: rvs data to calculated statistic value
        """
        raise NotImplementedError("Method is not implemented")


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
        pass

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
