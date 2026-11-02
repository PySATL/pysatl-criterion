"""Independent Gumbel fits and entropy references for Weibull KL."""

import numpy as np
import pytest
from scipy.stats import gumbel_l, gumbel_r

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as w


def criterion():
    return w.KullbackLeiblerWeibullGofStatistic(Weibull.DEFAULT.parse({}))


def entropy(y, m):
    ordered = np.sort(y)
    extended = np.pad(ordered, m, mode="edge")
    return np.mean(
        [np.log(len(y) * (extended[i + 2 * m] - extended[i]) / (2 * m)) for i in range(len(y))]
    )


@pytest.mark.parametrize("n,m", [(3, 1), (5, 2), (20, 1), (20, 3), (20, 4), (51, 7)])
def test_independent_gumbel_density_and_reflection(n, m):
    sample = np.random.default_rng(42).weibull(1.7, n)
    y = np.log(sample)
    loc, scale = gumbel_l.fit(y)
    expected = -entropy(y, m) - np.mean(gumbel_l.logpdf(y, loc=loc, scale=scale))
    assert criterion().execute_statistic(sample, m=m) == pytest.approx(expected, abs=1e-10)
    loc_r, scale_r = gumbel_r.fit(-y)
    reflected = -entropy(-y, m) - np.mean(gumbel_r.logpdf(-y, loc=loc_r, scale=scale_r))
    assert reflected == pytest.approx(expected, abs=1e-10)
    z = (y - loc) / scale
    assert np.mean(np.exp(z)) == pytest.approx(1, abs=1e-10)
    # Scalar translation of EWGoF's inner KL branch at a matched window.
    ordered = sorted(z)
    ew = (
        -sum(np.log(ordered[min(i + m, n - 1)] - ordered[max(i - m, 0)]) for i in range(n)) / n
        - np.mean(z)
        + np.mean(np.exp(z))
        - np.log(n / (2 * m))
    )
    assert criterion().execute_statistic(sample, m=m) == pytest.approx(ew, abs=1e-10)


def test_default_window_differs_from_ewgof():
    sample = np.random.default_rng(3).weibull(2, 20)
    test = criterion()
    assert test.execute_statistic(sample) == test.execute_statistic(sample, m=4)
    assert test.execute_statistic(sample) != pytest.approx(test.execute_statistic(sample, m=3))


@pytest.mark.parametrize("a,b", [(0.1, 1), (1, 0.2), (10, 3)])
def test_invariance_and_input(a, b):
    sample = np.array([1.4, 0.3, 1, 0.6, 2.2])
    before = sample.copy()
    test = criterion()
    assert test.execute_statistic(a * sample**b, m=1) == pytest.approx(
        test.execute_statistic(sample, m=1)
    )
    np.testing.assert_array_equal(sample, before)
    assert isinstance(test.alternative(), RightAlternative)


def test_ties_depend_on_window():
    sample = [1, 1, 1, 2, 3]
    assert criterion().execute_statistic(sample, m=1) == float("inf")
    assert criterion().execute_statistic([1, 1, 2, 3, 4], m=2) < float("inf")


@pytest.mark.parametrize("m", [0, -1, True, np.bool_(False), 1.5, 3, np.inf])
def test_invalid_windows(m):
    with pytest.raises(ValueError):
        criterion().execute_statistic([1, 2, 3, 4, 5], m=m)


@pytest.mark.parametrize("sample", [[1, 2], [1, 1, 1], [0, 1, 2], [1, 2, np.inf]])
def test_invalid_samples(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
