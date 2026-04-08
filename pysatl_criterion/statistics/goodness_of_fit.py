from abc import ABC

from typing_extensions import override

from pysatl_criterion.statistics.models import AbstractStatistic


class AbstractGoodnessOfFitStatistic(AbstractStatistic, ABC):
    """
    Abstract base class for goodness-of-fit statistics.
    """

    @staticmethod
    @override
    def code():
        """
        Get unique code identifier for goodness-of-fit statistics.

        :return: string code "GOODNESS_OF_FIT".
        """
        return "GOODNESS_OF_FIT"
