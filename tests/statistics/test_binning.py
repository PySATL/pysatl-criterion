"""Binning contracts, tail conservation and independent frequency references."""

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.statistics.goodness_of_fit.binning import (
    AlternativeOptimizedBins,
    DiscreteCategoryBins,
    EqualProbabilityBins,
    EqualWidthBins,
    ExplicitBins,
    FixedCount,
    FrequencyTable,
    InterpolatedBins,
    PowerCount,
    SqrtCount,
    SturgesCount,
)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (FixedCount(7), 7),
        (SturgesCount(), 8),
        (SqrtCount(), 10),
        (PowerCount(), 13),
    ],
)
def test_count_rules(rule, expected):
    assert rule.resolve(100) == expected
    with pytest.raises(ValueError):
        rule.resolve(0)


@pytest.mark.parametrize("count", [0, 1, True, 1.5, "auto"])
def test_invalid_count(count):
    with pytest.raises((TypeError, ValueError)):
        EqualProbabilityBins(count)


@pytest.mark.parametrize(
    "strategy",
    [
        EqualProbabilityBins(4),
        EqualWidthBins(2, 10, 4),
        ExplicitBins((2, 4, 6, 8, 10)),
        InterpolatedBins(2, 10, 4, kappa=0.4),
    ],
)
def test_uniform_equivalent_partitions(strategy):
    sample = [2.2, 3.5, 5, 7, 9, 10]
    table = strategy.build(sample, stats.uniform(2, 8))
    np.testing.assert_array_equal(table.observed, [2, 1, 1, 2])
    np.testing.assert_allclose(table.expected, [1.5] * 4)
    assert not table.observed.flags.writeable
    assert not table.expected.flags.writeable


def test_equal_probability_normal_without_ppf():
    # Inverse CDF is used only in the independent reference sample.
    sample = stats.norm.ppf([0.05, 0.2, 0.3, 0.55, 0.6, 0.95])
    table = EqualProbabilityBins(4).build(sample, stats.norm())
    np.testing.assert_array_equal(table.observed, [2, 1, 2, 1])
    np.testing.assert_allclose(table.expected, 1.5)


def test_equal_width_preserves_both_tails():
    table = EqualWidthBins(-1, 1, 2).build([-4, -0.5, 0.5, 4], stats.norm())
    np.testing.assert_array_equal(table.observed, [1, 1, 1, 1])
    np.testing.assert_allclose(
        table.expected, 4 * np.diff(stats.norm.cdf([-np.inf, -1, 0, 1, np.inf]))
    )


def test_upper_tail_precision():
    table = ExplicitBins((-np.inf, 8, 9, np.inf)).build([0, 8.5, 10], stats.norm())
    assert np.all(table.expected > 0)
    assert table.expected[-1] == pytest.approx(3 * stats.norm.sf(9), rel=1e-12, abs=0)


def test_interpolation_uses_null_quantiles():
    distribution = stats.expon()
    strategy = InterpolatedBins(0, 4, 2, kappa=0.5)
    # Two central bins plus the upper tail: three cells, two interior edges.
    edges = [0, 0.5 * stats.expon.ppf(1 / 3) + 1, 0.5 * stats.expon.ppf(2 / 3) + 2, np.inf]
    sample = [0.1, 0.5, 1.5, 3, 10]
    table = strategy.build(sample, distribution)
    reference = ExplicitBins(tuple(edges)).build(sample, distribution)
    np.testing.assert_array_equal(table.observed, reference.observed)
    np.testing.assert_allclose(table.expected, reference.expected)


@pytest.mark.parametrize("kappa", [0, 1])
def test_interpolation_endpoints(kappa):
    sample = [-2, 0.2, 0.7, 3]
    strategy = InterpolatedBins(-1, 1, 3, kappa=kappa)
    reference = EqualProbabilityBins(3) if kappa == 0 else EqualWidthBins(-1, 1, 3)
    actual, expected = strategy.build(sample, stats.norm()), reference.build(sample, stats.norm())
    np.testing.assert_array_equal(actual.observed, expected.observed)
    np.testing.assert_allclose(actual.expected, expected.expected)


def test_min_expected_and_max_bins():
    sample = np.linspace(0.01, 0.99, 20)
    table = EqualProbabilityBins(100, min_expected=5).build(sample, stats.uniform())
    assert len(table.expected) == 4
    assert np.all(table.expected == 5)
    table = EqualProbabilityBins(100, max_bins=3).build(sample, stats.uniform())
    assert len(table.expected) == 3
    with pytest.raises(ValueError, match="Insufficient"):
        EqualProbabilityBins(4, min_expected=5).build([0.5] * 9, stats.uniform())


def test_merge_depends_only_on_expected_counts():
    strategy = ExplicitBins((0, 0.01, 0.5, 1), min_expected=5)
    first = strategy.build([0.1] * 20, stats.uniform())
    second = strategy.build([0.9] * 20, stats.uniform())
    np.testing.assert_allclose(first.expected, [10, 10])
    np.testing.assert_array_equal(first.expected, second.expected)
    np.testing.assert_array_equal(first.observed, [20, 0])
    np.testing.assert_array_equal(second.observed, [0, 20])


def test_discrete_boundary_atoms():
    table = DiscreteCategoryBins((1, 3)).build([1, 1, 2, 3, 4, 5], stats.geom(0.5))
    np.testing.assert_array_equal(table.observed, [2, 2, 2])
    np.testing.assert_allclose(table.expected, 6 * np.array([0.5, 0.375, 0.125]))


@pytest.mark.parametrize("edges", [(0, 0, 1), (0, np.nan, 1), (0, 1)])
def test_invalid_edges(edges):
    with pytest.raises(ValueError):
        ExplicitBins(edges)


def test_missing_tails_and_outside_sample_are_rejected():
    with pytest.raises(ValueError, match="entire null"):
        ExplicitBins((-1, 0, 1)).build([0], stats.norm())
    with pytest.raises(ValueError, match="outside"):
        ExplicitBins((0, 0.5, 1)).build([2], stats.uniform())
    with pytest.raises(ValueError, match="zero-probability"):
        ExplicitBins((-1, 0, 0.5, 1)).build([-0.5, 0.7], stats.uniform())


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ([1, 2], [1, 1]),
        ([1, 1], [0, 2]),
        ([1], [1]),
        ([np.nan, 2], [1, 2]),
    ],
)
def test_invalid_frequency_table(observed, expected):
    with pytest.raises(ValueError):
        FrequencyTable(observed, expected)


def test_optimized_selection_is_reproducible_and_fixed_before_testing():
    kwargs = {
        "candidates": [EqualProbabilityBins(2), EqualProbabilityBins(4)],
        "distribution": stats.uniform(),
        "null_sampler": lambda n, rng: rng.uniform(size=n),
        "alternative_sampler": lambda n, rng: rng.beta(1, 4, size=n),
        "kernel": lambda o, e: float(np.sum((o - e) ** 2 / e)),
        "sample_size": 40,
        "repetitions": 40,
        "seed": 19,
    }
    first = AlternativeOptimizedBins.select(**kwargs)
    second = AlternativeOptimizedBins.select(**kwargs)
    assert first == second
    assert first.selected in kwargs["candidates"]
    assert all(0 <= p <= 1 for p in first.estimated_powers)
    table = first.build(np.linspace(0, 1, 40), stats.uniform())
    assert table.observed.sum() == 40
    with pytest.raises(ValueError, match="sample size"):
        first.build([0.2], stats.uniform())
