from enum import Enum

from src.pysatl_criterion.statistics import (
    AbstractBetaGofStatistic,
    AbstractExponentialityGofStatistic,
    AbstractGammaGofStatistic,
    AbstractLogNormalGofStatistic,
    AbstractNormalityGofStatistic,
    AbstractStudentGofStatistic,
    AbstractUniformGofStatistic,
    AbstractWeibullGofStatistic,
)
from src.pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic


class DistributionType(Enum):
    """
    Distribution type.
    """

    base_class: type[AbstractGoodnessOfFitStatistic]

    NORMAL = ("normal", AbstractNormalityGofStatistic)
    EXPONENTIAL = ("exponential", AbstractExponentialityGofStatistic)
    WEIBULL = ("weibull", AbstractWeibullGofStatistic)
    UNIFORM = ("uniform", AbstractUniformGofStatistic)
    STUDENT = ("student", AbstractStudentGofStatistic)
    GAMMA = ("gamma", AbstractGammaGofStatistic)
    BETA = ("beta", AbstractBetaGofStatistic)
    LOG_NORMAL = ("log_normal", AbstractLogNormalGofStatistic)

    def __new__(cls, value: str, base_class: type[AbstractGoodnessOfFitStatistic]):
        obj = object.__new__(cls)
        # The first item in the tuple becomes the canonical .value
        obj._value_ = value
        obj.base_class = base_class
        return obj

    @classmethod
    def list(cls):
        """
        Get a list of all distribution string identifiers.

        :return: list of string values for all members in the enum.
        """
        return [member.value for member in cls]
