"""Composable category construction for frequency-based goodness-of-fit tests."""

from pysatl_criterion.statistics.goodness_of_fit.binning.base import BinningStrategy, FrequencyTable
from pysatl_criterion.statistics.goodness_of_fit.binning.continuous import (
    EqualProbabilityBins,
    EqualWidthBins,
    ExplicitBins,
    InterpolatedBins,
)
from pysatl_criterion.statistics.goodness_of_fit.binning.counts import (
    CountRule,
    FixedCount,
    PowerCount,
    SqrtCount,
    SturgesCount,
)
from pysatl_criterion.statistics.goodness_of_fit.binning.discrete import DiscreteCategoryBins
from pysatl_criterion.statistics.goodness_of_fit.binning.optimized import AlternativeOptimizedBins


__all__ = [
    "AlternativeOptimizedBins",
    "BinningStrategy",
    "CountRule",
    "DiscreteCategoryBins",
    "EqualProbabilityBins",
    "EqualWidthBins",
    "ExplicitBins",
    "FixedCount",
    "FrequencyTable",
    "InterpolatedBins",
    "PowerCount",
    "SqrtCount",
    "SturgesCount",
]
