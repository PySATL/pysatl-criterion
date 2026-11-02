"""Stable public names for built-in binning policies, not Python repr strings."""

import math

from pysatl_criterion.statistics.configuration import normalize_configuration


def count_configuration(count):
    from pysatl_criterion.statistics.goodness_of_fit.binning.counts import (
        FixedCount,
        PowerCount,
        SqrtCount,
        SturgesCount,
    )

    if isinstance(count, int) and not isinstance(count, bool):
        count = FixedCount(count)
    names = {FixedCount: "fixed", PowerCount: "power", SqrtCount: "sqrt", SturgesCount: "sturges"}
    if type(count) not in names:
        raise ValueError("Custom count rules must implement their own stable to_config()")
    result: dict = {"type": names[type(count)], "version": 1}
    if type(count) is FixedCount:
        result["count"] = count.count
    if type(count) is PowerCount:
        result.update(coefficient=count.coefficient, exponent=count.exponent)
    return normalize_configuration(result)


def binning_configuration(strategy):
    from pysatl_criterion.statistics.goodness_of_fit.binning.continuous import (
        EqualProbabilityBins,
        EqualWidthBins,
        ExplicitBins,
        InterpolatedBins,
    )
    from pysatl_criterion.statistics.goodness_of_fit.binning.counts import FixedCount
    from pysatl_criterion.statistics.goodness_of_fit.binning.discrete import DiscreteCategoryBins
    from pysatl_criterion.statistics.goodness_of_fit.binning.optimized import (
        AlternativeOptimizedBins,
    )

    names = {
        EqualProbabilityBins: "equal_probability",
        EqualWidthBins: "equal_width",
        ExplicitBins: "explicit",
        InterpolatedBins: "interpolated",
        DiscreteCategoryBins: "discrete_categories",
        AlternativeOptimizedBins: "selected",
    }
    if type(strategy) not in names:
        raise ValueError("Custom binning strategies must implement their own stable to_config()")
    result: dict = {"type": names[type(strategy)], "version": 1}
    if type(strategy) is AlternativeOptimizedBins:
        result.update(
            selected=_custom_configuration(strategy.selected), sample_size=strategy.sample_size
        )
        return normalize_configuration(result)
    result["min_expected"] = strategy.min_expected
    if hasattr(strategy, "count"):
        from numbers import Integral

        rule = (
            FixedCount(strategy.count) if isinstance(strategy.count, Integral) else strategy.count
        )
        result.update(count=_custom_configuration(rule), max_bins=strategy.max_bins)
    if isinstance(strategy, EqualWidthBins):
        result.update(lower=strategy.lower, upper=strategy.upper)
    if type(strategy) is InterpolatedBins:
        result["kappa"] = strategy.kappa
    if type(strategy) is ExplicitBins:
        result["edges"] = [
            "-inf" if x == -math.inf else "+inf" if x == math.inf else x for x in strategy.edges
        ]
    if type(strategy) is DiscreteCategoryBins:
        result["cut_points"] = list(strategy.cut_points)
    return normalize_configuration(result)


def _custom_configuration(value) -> dict:
    serialize = getattr(value, "to_config", None)
    if not callable(serialize):
        raise TypeError("Storage calibration requires a stable to_config() dictionary")
    result = serialize()
    if not isinstance(result, dict):
        raise TypeError("to_config() must return a dictionary")
    return result
