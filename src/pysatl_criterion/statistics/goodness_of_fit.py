from abc import ABC, abstractmethod

from typing_extensions import override

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.statistics.models import AbstractStatistic


class AbstractGoodnessOfFitStatistic(AbstractStatistic, ABC):
    """
    Abstract base class for goodness-of-fit statistics.
    """

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
