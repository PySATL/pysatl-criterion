"""Independent Taylor derivatives and cumulative-spacing identity for TS."""

import numpy as np
import pytest

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import TwoSidedAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as w


def criterion():
    return w.TikuSinghWeibullGofStatistic(Weibull.DEFAULT.parse({}))


@pytest.mark.parametrize("n", [3, 5, 20, 100])
def test_cumulative_spacing_reference(n):
    p = np.arange(1, n + 1) / (n + 1)
    q = 1 - p
    ell = -np.log(q)
    # Analytic derivatives, independent of GoFNS's derivative recurrence.
    d2 = (ell - 1) / (q**2 * ell**2)
    d3 = (2 * ell**2 - 3 * ell + 2) / (q**3 * ell**3)
    d4 = (6 * ell**3 - 11 * ell**2 + 12 * ell - 6) / (q**4 * ell**4)
    means = (
        np.log(ell)
        + p * q * d2 / (2 * (n + 2))
        + p * q / (n + 2) ** 2 * ((q - p) * d3 / 3 + p * q * d4 / 8)
    )
    sample = np.random.default_rng(132).weibull(1.7, n)
    gaps = np.diff(np.log(np.sort(sample))) / np.diff(means)
    expected = 2 * np.mean(np.cumsum(gaps)[:-1] / gaps.sum())
    result = criterion().execute_statistic(sample)
    assert result == pytest.approx(expected, rel=1e-13)
    assert 0 < result < 2


@pytest.mark.parametrize("a,b", [(1, 0.2), (1, 5), (0.01, 1), (100, 2)])
def test_invariance_and_input_preservation(a, b):
    sample = np.array([2.2, 0.3, 1, 0.6, 1.4])
    before = sample.copy()
    test = criterion()
    assert test.execute_statistic(a * sample**b) == pytest.approx(test.execute_statistic(sample))
    np.testing.assert_array_equal(sample, before)
    assert isinstance(test.alternative(), TwoSidedAlternative)


@pytest.mark.parametrize("sample,expected", [([1, 1, 2], 0), ([1, 2, 2], 2)])
def test_three_point_boundary(sample, expected):
    assert criterion().execute_statistic(sample) == pytest.approx(expected)


@pytest.mark.parametrize("sample", [[1, 2], [1, 1, 1], [0, 1, 2], [1, 2, np.inf]])
def test_invalid_sample(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
