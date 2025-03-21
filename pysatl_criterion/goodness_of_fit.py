from abc import ABC

from typing_extensions import override

from pysatl_criterion.models import AbstractStatistic


class AbstractGoodnessOfFitStatistic(AbstractStatistic, ABC):
    @staticmethod
    @override
    def code():
        return "GOODNESS_OF_FIT"
