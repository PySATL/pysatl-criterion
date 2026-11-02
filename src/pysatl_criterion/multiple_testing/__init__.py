"""Typed multiple testing: ``Holm().test(PValues(p), alpha=0.05)``.

The package root and api.testing export the same typed procedures.
See docs/multiple_testing/index.md for migration from the removed classmethod API.
"""

from collections.abc import Mapping
from types import MappingProxyType

from .base import AbstractMultipleTesting
from .inputs import PValues, ResamplingPValues, ResamplingStatistics
from .metadata import ProcedureMetadata
from .pvalue import (
    BenjaminiHochberg,
    BenjaminiKriegerYekutieli,
    BenjaminiYekutieli,
    Bonferroni,
    Hochberg,
    HollandCopenhaver,
    Holm,
    HolmSidak,
    Hommel,
    Rom,
    Sidak,
    Simes,
    TwoStageBenjaminiHochberg,
)
from .pvalue.base import AbstractAlphaDependentPValueProcedure, AbstractPValueAdjustment
from .resampling import JointResamplingAssumptions, WestfallYoungMaxT, WestfallYoungMinP
from .resampling.base import AbstractResamplingMultipleTesting, AbstractResamplingPValueTesting
from .result import GlobalTestResult, MultipleTestingResult


P_VALUE_PROCEDURES: Mapping[str, type[AbstractMultipleTesting[PValues]]] = MappingProxyType(
    {
        method.short_code(): method
        for method in (
            Rom,
            HollandCopenhaver,
            Bonferroni,
            Sidak,
            Holm,
            HolmSidak,
            Hochberg,
            Hommel,
            BenjaminiHochberg,
            BenjaminiYekutieli,
            TwoStageBenjaminiHochberg,
            BenjaminiKriegerYekutieli,
        )
    }
)

GLOBAL_TESTS: Mapping[str, type[Simes]] = MappingProxyType({"simes": Simes})
RESAMPLING_STATISTIC_PROCEDURES: Mapping[str, type[AbstractResamplingMultipleTesting]] = (
    MappingProxyType({WestfallYoungMaxT.short_code(): WestfallYoungMaxT})
)
RESAMPLING_PVALUE_PROCEDURES: Mapping[str, type[AbstractResamplingPValueTesting]] = (
    MappingProxyType({WestfallYoungMinP.short_code(): WestfallYoungMinP})
)

__all__ = [
    "GLOBAL_TESTS",
    "P_VALUE_PROCEDURES",
    "RESAMPLING_PVALUE_PROCEDURES",
    "RESAMPLING_STATISTIC_PROCEDURES",
    "AbstractAlphaDependentPValueProcedure",
    "AbstractMultipleTesting",
    "AbstractPValueAdjustment",
    "AbstractResamplingMultipleTesting",
    "AbstractResamplingPValueTesting",
    "BenjaminiHochberg",
    "BenjaminiKriegerYekutieli",
    "BenjaminiYekutieli",
    "Bonferroni",
    "GlobalTestResult",
    "Hochberg",
    "HollandCopenhaver",
    "Holm",
    "HolmSidak",
    "Hommel",
    "JointResamplingAssumptions",
    "MultipleTestingResult",
    "PValues",
    "ProcedureMetadata",
    "ResamplingPValues",
    "ResamplingStatistics",
    "Rom",
    "Sidak",
    "Simes",
    "TwoStageBenjaminiHochberg",
    "WestfallYoungMaxT",
    "WestfallYoungMinP",
]
