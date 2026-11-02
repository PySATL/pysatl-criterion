from abc import ABC, abstractmethod

from pysatl_criterion.statistics.alternative import Alternative
from pysatl_criterion.statistics.hypothesis import (
    Hypothesis,
)


class AbstractStatistic(ABC):
    @abstractmethod
    def hypothesis(self) -> Hypothesis:
        """
        Get alternative type.

        :return: alternative type.
        """

    @abstractmethod
    def alternative(self) -> Alternative:
        """
        Get alternative.

        :return: alternative.
        """

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
