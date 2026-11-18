"""Independent expected-order expansion and split checks for MSF."""

import numpy as np
import pytest

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as w


def criterion():
    return w.MannScheuerFertigWeibullGofStatistic(Weibull.DEFAULT.parse({}))


@pytest.mark.parametrize("n", [3, 4, 5, 6, 20, 21, 100])
def test_independent_reference(n):
    p = np.arange(1, n + 1) / (n + 1)
    q = 1 - p
    ell = -np.log(q)
    # Independent analytic derivatives, rather than GoFNS's recurrence.
    d2 = (ell - 1) / (q**2 * ell**2)
    d3 = (2 * ell**2 - 3 * ell + 2) / (q**3 * ell**3)
    d4 = (6 * ell**3 - 11 * ell**2 + 12 * ell - 6) / (q**4 * ell**4)
    means = (
        np.log(ell)
        + p * q * d2 / (2 * (n + 2))
        + p * q / (n + 2) ** 2 * ((q - p) * d3 / 3 + p * q * d4 / 8)
    )
    sample = np.random.default_rng(17).weibull(1.5, n)
    h = np.diff(np.log(np.sort(sample))) / np.diff(means)
    # Complement of the lower cumulative fraction, not the production slice.
    expected = 1 - sum(h[i - 1] for i in range(1, n // 2 + 1)) / sum(h)
    result = criterion().execute_statistic(sample)
    assert result == pytest.approx(expected, rel=1e-13)
    assert 0 < result < 1


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_split_boundary(n):
    r = n // 2
    for gap in range(1, n):
        # A single nonzero log gap on either side of the split.
        sample = [1.0] * gap + [2.0] * (n - gap)
        assert criterion().execute_statistic(sample) == pytest.approx(float(gap > r))


@pytest.mark.parametrize("a,b", [(0.1, 1), (1, 0.2), (10, 3)])
def test_invariance_and_input(a, b):
    sample = np.array([1.4, 0.3, 1, 0.6, 2.2])
    before = sample.copy()
    test = criterion()
    assert test.execute_statistic(a * sample**b) == pytest.approx(test.execute_statistic(sample))
    np.testing.assert_array_equal(sample, before)
    assert isinstance(test.alternative(), RightAlternative)


def test_public_name_and_short_code():
    cls = type(criterion())
    assert cls.__name__ == "MannScheuerFertigWeibullGofStatistic"
    assert cls.__name__ in w.__all__
    assert cls.short_code() == "MSF"
    old_name = "MSF" + "WeibullGofStatistic"
    assert not hasattr(w, old_name)
    assert old_name not in w.__all__


@pytest.mark.parametrize("sample", [[1, 2], [1, 1, 1], [0, 1, 2], [1, 2, np.inf]])
def test_invalid_samples(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
