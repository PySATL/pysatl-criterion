from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("pysatl-criterion")
except PackageNotFoundError:
    __version__ = "0+unknown"

from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator

from .distribution.distribution_kind import DistributionKind
from .distribution.distribution_parameter_descriptor import DistributionParameterDescriptor
from .distribution.distribution_type import DistributionType
from .hypothesis_testing.critical_values.calculator.model import CriticalValueCalculator
from .hypothesis_testing.goodness_of_fit_test.goodness_of_fit_test import GoodnessOfFitTest
from .multiple_testing import (
    BenjaminiHochberg,
    BenjaminiKriegerYekutieli,
    BenjaminiYekutieli,
    Bonferroni,
    GlobalTestResult,
    Hochberg,
    HollandCopenhaver,
    Holm,
    HolmSidak,
    Hommel,
    JointResamplingAssumptions,
    MultipleTestingResult,
    PValues,
    Rom,
    Sidak,
    Simes,
    TwoStageBenjaminiHochberg,
    WestfallYoungMaxT,
    WestfallYoungMinP,
)


__all__ = [
    "BenjaminiHochberg",
    "BenjaminiKriegerYekutieli",
    "BenjaminiYekutieli",
    "Bonferroni",
    "CriticalValueCalculator",
    "DistributionKind",
    "DistributionParameterDescriptor",
    "DistributionType",
    "GlobalTestResult",
    "GoodnessOfFitTest",
    "Hochberg",
    "HollandCopenhaver",
    "Holm",
    "HolmSidak",
    "Hommel",
    "JointResamplingAssumptions",
    "MultipleTestingResult",
    "PValueCalculator",
    "PValues",
    "Rom",
    "Sidak",
    "Simes",
    "TwoStageBenjaminiHochberg",
    "WestfallYoungMaxT",
    "WestfallYoungMinP",
]
