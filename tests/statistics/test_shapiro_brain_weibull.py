"""Shapiro-Brain checks against Shell (2003), equation (9) and Table 1."""

from math import fsum, log

import numpy as np
import pytest

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import TwoSidedAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    ShapiroBrainWeibullGofStatistic,
)


SHELL_SAMPLE = np.array(
    [
        2,
        3.25,
        4.1,
        5.6,
        6.05,
        8.01,
        8.9,
        9.25,
        9.4,
        9.8,
        10.1,
        10.6,
        12,
        12.5,
        12.9,
        13.3,
        13.6,
        14,
        15,
        15.5,
        16.9,
        18,
        18.5,
        20,
    ]
)


def shell_reference(sample):
    """Scalar transcription; return raw W and analytically centered W."""
    y = sorted(log(float(x)) for x in sample)
    n = len(y)
    mean = fsum(y) / n
    w = [log((n + 1) / (n + 1 - i)) for i in range(1, n)]
    v = [wi * (1 + log(wi)) - 1 for wi in w]
    w.append(n - fsum(w))
    v.append(0.4228 * n - fsum(v))
    l1 = fsum(wi * yi for wi, yi in zip(w, y, strict=True))
    l2 = fsum(vi * yi for vi, yi in zip(v, y, strict=True))
    b = (0.6079 * l2 - 0.2570 * l1) / n
    variance_sum = fsum((yi - mean) ** 2 for yi in y)
    centered_b = b - 0.00002012 * mean
    return n * b**2 / variance_sum, n * centered_b**2 / variance_sum


def criterion():
    return ShapiroBrainWeibullGofStatistic(Weibull.DEFAULT.parse({}))


def test_published_example_and_centering_adjustment():
    raw, centered = shell_reference(SHELL_SAMPLE)
    assert raw == pytest.approx(0.4930, abs=5e-5)
    assert centered == pytest.approx(0.49292565934030375, abs=1e-13)
    assert criterion().execute_statistic(SHELL_SAMPLE) == pytest.approx(centered, abs=1e-13)
    assert abs(raw - centered) > 1e-4
    assert isinstance(criterion().alternative(), TwoSidedAlternative)


@pytest.mark.parametrize("sample", [[0.3, 0.6, 1, 1.4, 2.2], [1, 1, 2, 3], [1e-200, 1, 1e200]])
def test_independent_scalar_reference(sample):
    assert criterion().execute_statistic(sample) == pytest.approx(shell_reference(sample)[1])


@pytest.mark.parametrize("scale", [1e-100, 1, 1e100])
@pytest.mark.parametrize("power", [0.2, 1, 3])
def test_log_affine_invariance_and_no_mutation(scale, power):
    sample = scale * SHELL_SAMPLE[::-1] ** power
    original = sample.copy()
    expected = criterion().execute_statistic(SHELL_SAMPLE)
    assert criterion().execute_statistic(sample) == pytest.approx(expected, rel=1e-11)
    np.testing.assert_array_equal(sample, original)


def test_two_observations_have_no_discriminating_power():
    assert criterion().execute_statistic([1, 2]) == pytest.approx(
        criterion().execute_statistic([0.001, 100])
    )


@pytest.mark.parametrize("sample", [[], [1], [1, 1], [0, 1], [-1, 2], [np.inf, 1], [np.nan, 1]])
def test_invalid_samples(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
