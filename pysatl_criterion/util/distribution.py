from enum import Enum

from pysatl_criterion.statistics import (
    AbstractExponentialityGofStatistic,
    AbstractGammaGofStatistic,
    AbstractNormalityGofStatistic,
    AbstractStudentGofStatistic,
    AbstractWeibullGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.uniform import AbstractUniformGofStatistic


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

    def __new__(cls, value: str, base_class: type[AbstractGoodnessOfFitStatistic]):
        obj = object.__new__(cls)
        # The first item in the tuple becomes the canonical .value
        obj._value_ = value
        obj.base_class = base_class
        return obj

    @classmethod
    def list(cls):
        """
        Collect all enum values.

        @return: enum values
        """
        return [member.value for member in cls]
