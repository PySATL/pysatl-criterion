"""Check Krit (2014), (3.12), through an independent regression residual."""

import numpy as np
import pytest
from scipy.stats import gumbel_l

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    SmithBainWeibullGofStatistic,
)


def criterion():
    return SmithBainWeibullGofStatistic(Weibull.DEFAULT.parse({}))


def regression_reference(sample):
    y = np.sort(np.log(sample))
    n = len(y)
    scores = gumbel_l.ppf(np.arange(1, n + 1) / (n + 1))
    design = np.column_stack((np.ones(n), scores))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return n * np.dot(residual, residual) / np.sum((y - y.mean()) ** 2)


@pytest.mark.parametrize("sample", [[0.3, 0.6, 1, 1.4, 2.2], [1, 1, 2, 3], [1e-200, 1, 1e200]])
def test_regression_residual_reference(sample):
    value = criterion().execute_statistic(sample)
    assert value == pytest.approx(regression_reference(sample), abs=1e-12)
    assert 0 <= value <= len(sample)
    assert isinstance(criterion().alternative(), RightAlternative)


@pytest.mark.parametrize("scale", [1e-100, 1, 1e100])
@pytest.mark.parametrize("power", [0.2, 1, 3])
def test_log_affine_invariance_and_no_mutation(scale, power):
    base = np.array([0.3, 0.6, 1, 1.4, 2.2])
    sample = scale * base[::-1] ** power
    original = sample.copy()
    assert criterion().execute_statistic(sample) == pytest.approx(
        regression_reference(base), abs=1e-11
    )
    np.testing.assert_array_equal(sample, original)


def test_exact_probability_plot_and_two_observation_degeneracy():
    probabilities = np.arange(1, 7) / 7
    sample = 2 * (-np.log1p(-probabilities)) ** (1 / 3)
    assert criterion().execute_statistic(sample) == pytest.approx(0, abs=1e-14)
    for pair in ([1, 2], [0.001, 100]):
        assert criterion().execute_statistic(pair) == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("sample", [[], [1], [1, 1], [0, 1], [-1, 2], [np.inf, 1], [np.nan, 1]])
def test_invalid_samples(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
