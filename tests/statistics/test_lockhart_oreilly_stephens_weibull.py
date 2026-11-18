"""Independent weighted EDF integral and boundary checks for spacing AD."""

from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as w


def criterion():
    return w.LockhartOReillyStephensWeibullGofStatistic(Weibull.DEFAULT.parse({}))


@pytest.mark.parametrize("n", [3, 4, 5, 20, 21, 100])
def test_independent_integral(n):
    p = np.arange(1, n + 1) / (n + 1)
    q = 1 - p
    ell = -np.log(q)
    # Analytic derivatives independent of the GoFNS recurrence.
    d2 = (ell - 1) / (q**2 * ell**2)
    d3 = (2 * ell**2 - 3 * ell + 2) / (q**3 * ell**3)
    d4 = (6 * ell**3 - 11 * ell**2 + 12 * ell - 6) / (q**4 * ell**4)
    means = (
        np.log(ell)
        + p * q * d2 / (2 * (n + 2))
        + p * q / (n + 2) ** 2 * ((q - p) * d3 / 3 + p * q * d4 / 8)
    )
    sample = np.random.default_rng(51).weibull(2, n)
    gaps = np.diff(np.log(np.sort(sample))) / np.diff(means)
    z = np.cumsum(gaps)[:-1] / sum(gaps)
    edges = np.r_[0, z, 1]
    size = n - 2
    expected = sum(
        quad(lambda u, j=j: size * (j / size - u) ** 2 / (u * (1 - u)), lo, hi)[0]
        for j, (lo, hi) in enumerate(pairwise(edges))
    )
    assert criterion().execute_statistic(sample) == pytest.approx(expected, rel=1e-11)
    assert expected >= 0


@pytest.mark.parametrize("sample", [[1, 1, 2], [1, 2, 2], [1, 1, 2, 3], [1, 2, 3, 3]])
def test_endpoint_ties(sample):
    assert criterion().execute_statistic(sample) == float("inf")


def test_interior_ties():
    assert np.isfinite(criterion().execute_statistic([1, 2, 2, 3]))


def test_reverse_sum_preserves_small_survival(monkeypatch):
    # Direct 1-cumsum(g) rounds to zero, but the final gap is positive.
    g = np.array([0.5, 0.5, 1e-20])
    test = criterion()
    monkeypatch.setattr(test, "_spacings", lambda sample: g)
    assert np.cumsum(g)[-2] == 1
    expected = -2 - (np.log(0.5) + np.log(1e-20) + 3 * np.log(0.5)) / 2
    assert test.execute_statistic([1, 2, 3, 4]) == pytest.approx(expected)


@pytest.mark.parametrize("a,b", [(0.1, 1), (1, 0.2), (10, 3)])
def test_invariance_and_input(a, b):
    sample = np.array([1.4, 0.3, 1, 0.6, 2.2])
    before = sample.copy()
    test = criterion()
    assert test.execute_statistic(a * sample**b) == pytest.approx(test.execute_statistic(sample))
    np.testing.assert_array_equal(sample, before)
    assert isinstance(test.alternative(), RightAlternative)


def test_public_name():
    cls = type(criterion())
    assert cls.__name__ == "LockhartOReillyStephensWeibullGofStatistic"
    assert cls.__name__ in w.__all__
    assert cls.short_code() == "LOS"
    old_name = "LOS" + "WeibullGofStatistic"
    assert not hasattr(w, old_name)
    assert old_name not in w.__all__


@pytest.mark.parametrize("sample", [[1, 2], [1, 1, 1], [0, 1, 2], [1, 2, np.inf]])
def test_invalid_samples(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
