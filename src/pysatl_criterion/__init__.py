from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("pysatl-criterion")
except PackageNotFoundError:
    __version__ = "0+unknown"

from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator

from .distribution.distribution_type import DistributionParameterDescriptor, DistributionType
from .hypothesis_testing.critical_values.calculator.model import CriticalValueCalculator
from .hypothesis_testing.goodness_of_fit_test.goodness_of_fit_test import GoodnessOfFitTest
from .hypothesis_testing.multiple_testing.fdr import BenjaminiYekutieli
from .hypothesis_testing.multiple_testing.fwer import (
    BonferroniMultipleTesting,
    Holm,
    SidakHolm,
    SidakMultipleTesting,
)


__all__ = [
    "GoodnessOfFitTest",
    "DistributionType",
    "DistributionParameterDescriptor",
    "BonferroniMultipleTesting",
    "Holm",
    "SidakMultipleTesting",
    "SidakHolm",
    "BenjaminiYekutieli",
    "PValueCalculator",
    "CriticalValueCalculator",
]
