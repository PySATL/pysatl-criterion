from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("pysatl-criterion")
except PackageNotFoundError:
    __version__ = "0+unknown"

from .hypothesis_testing.critical_values.resolver.model import CriticalValueResolver
from .hypothesis_testing.goodness_of_fit_test.goodness_of_fit_test import GoodnessOfFitTest
from .hypothesis_testing.model import TestMethod
from .hypothesis_testing.multiple_testing.fdr import BenjaminiYekutieli
from .hypothesis_testing.multiple_testing.fwer import (
    BonferroniMultipleTesting,
    Holm,
    SidakHolm,
    SidakMultipleTesting,
)
from .hypothesis_testing.p_value.resolver.model import PValueResolver
from .utils.distribution import DistributionType


__all__ = [
    "GoodnessOfFitTest",
    "TestMethod",
    "DistributionType",
    "BonferroniMultipleTesting",
    "Holm",
    "SidakMultipleTesting",
    "SidakHolm",
    "BenjaminiYekutieli",
    "PValueResolver",
    "CriticalValueResolver",
]
