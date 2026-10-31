"""Independent checks of Krit (2014), equations (3.16), (3.18)-(3.20)."""

from math import fsum, log, sqrt

import numpy as np
import pytest

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import TwoSidedAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    OzturkKorukogluWeibullGofStatistic,
)


def criterion():
    return OzturkKorukogluWeibullGofStatistic(Weibull.DEFAULT.parse({}))


def reference(sample):
    logs = sorted(log(float(x)) for x in sample)
    n = len(logs)
    w = [log((n + 1) / (n + 1 - i)) for i in range(1, n)]
    v = [wi * (1 + log(wi)) - 1 for wi in w]
    w.append(n - fsum(w))
    v.append(0.4228 * n - fsum(v))
    raw = fsum((0.6079 * vi - 0.2570 * wi) * yi for vi, wi, yi in zip(v, w, logs, strict=True)) / n
    b = raw - 0.00002012 * fsum(logs) / n
    s = fsum(logs[j] - logs[i] for j in range(n) for i in range(j)) / (n * (n - 1) * log(2))
    assert s > 0
    return (b / s - 1 - 0.13 / sqrt(n) + 1.18 / n) / (0.49 / sqrt(n) - 0.36 / n)


@pytest.mark.parametrize("sample", [[0.3, 0.6, 1, 1.4, 2.2], [1, 1, 2, 3], [1e-200, 1, 1e200]])
def test_independent_reference(sample):
    assert criterion().execute_statistic(sample) == pytest.approx(reference(sample), abs=1e-12)
    assert isinstance(criterion().alternative(), TwoSidedAlternative)


@pytest.mark.parametrize("scale", [1e-100, 1, 1e100])
@pytest.mark.parametrize("power", [0.2, 1, 3])
def test_log_affine_invariance_and_no_mutation(scale, power):
    base = np.array([0.3, 0.6, 1, 1.4, 2.2])
    sample = scale * base[::-1] ** power
    original = sample.copy()
    assert criterion().execute_statistic(sample) == pytest.approx(reference(base), abs=1e-11)
    np.testing.assert_array_equal(sample, original)


def test_two_observations_have_no_discriminating_power():
    assert criterion().execute_statistic([1, 2]) == pytest.approx(
        criterion().execute_statistic([0.001, 100])
    )


@pytest.mark.parametrize("sample", [[], [1], [1, 1], [0, 1], [-1, 2], [np.inf, 1], [np.nan, 1]])
def test_invalid_samples(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
