"""Blom linear-weight reference, independent of the least-squares implementation."""

from math import asin, exp, fsum, log, pi, sqrt

import numpy as np
import pytest
from scipy.stats import gumbel_l

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as w


def criterion():
    return w.ColesWeibullGofStatistic(Weibull.DEFAULT.parse({}))


def reference(sample):
    y = sorted(log(float(x)) for x in sample)
    n = len(y)
    theta, means = [0.0], [0.0]
    for i in range(1, n + 1):
        p = i / (n + 1)
        q = 1 - p
        ell = -log(q)
        # Explicit analytic derivatives of log(-log(1-p)).
        d2 = (ell - 1) / (q**2 * ell**2)
        d3 = (2 * ell**2 - 3 * ell + 2) / (q**3 * ell**3)
        d4 = (6 * ell**3 - 11 * ell**2 + 12 * ell - 6) / (q**4 * ell**4)
        means.append(
            log(ell)
            + p * q * d2 / (2 * (n + 2))
            + p * q / (n + 2) ** 2 * ((q - p) * d3 / 3 + p * q * d4 / 8)
        )
        theta.append(q * ell)
    theta.append(0.0)
    means.append(0.0)
    c1 = [theta[j] - theta[j + 1] for j in range(n + 1)]
    c2 = [theta[j] * means[j] - theta[j + 1] * means[j + 1] for j in range(n + 1)]
    a, b, c = (
        fsum(x * x for x in c1),
        fsum(x * z for x, z in zip(c1, c2, strict=True)),
        fsum(x * x for x in c2),
    )
    det = a * c - b * b
    g1 = [
        theta[i] * (c * (c1[i] - c1[i - 1]) - b * (c2[i] - c2[i - 1])) / det
        for i in range(1, n + 1)
    ]
    g2 = [
        theta[i] * (a * (c2[i] - c2[i - 1]) - b * (c1[i] - c1[i - 1])) / det
        for i in range(1, n + 1)
    ]
    assert fsum(g1) == pytest.approx(1)
    assert fsum(g2) == pytest.approx(0, abs=1e-14)
    location = fsum(a * b for a, b in zip(g1, y, strict=True))
    scale = fsum(a * b for a, b in zip(g2, y, strict=True))
    assert scale > 0
    z = [(v - location) / scale for v in y]
    return max(
        abs(2 / pi * (asin(sqrt((i - 0.5) / n)) - asin(sqrt(-np.expm1(-exp(v))))))
        for i, v in enumerate(z, 1)
    )


@pytest.mark.parametrize(
    "sample", [[0.3, 0.6, 1, 1.4, 2.2], [1, 1, 2, 3], [1e-200, 1, 1e200], [1, 2]]
)
def test_linear_weight_reference(sample):
    value = criterion().execute_statistic(sample)
    assert value == pytest.approx(reference(sample), abs=1e-12)
    assert 0 <= value <= 1
    assert isinstance(criterion().alternative(), RightAlternative)


@pytest.mark.parametrize("scale", [1e-100, 1, 1e100])
@pytest.mark.parametrize("power", [0.2, 1, 3])
def test_log_affine_invariance_and_no_mutation(scale, power):
    base = np.array([0.3, 0.6, 1, 1.4, 2.2])
    sample = scale * base[::-1] ** power
    before = sample.copy()
    assert criterion().execute_statistic(sample) == pytest.approx(reference(base), abs=1e-11)
    np.testing.assert_array_equal(sample, before)


def test_blom_replaces_mle_and_refits_each_sample(monkeypatch):
    sample = np.array([0.3, 0.6, 1, 1.4, 2.2])
    location, scale = gumbel_l.fit(np.log(sample))
    u = gumbel_l.cdf((np.log(sample) - location) / scale)
    old = np.max(
        np.abs(
            2 / np.pi * (np.arcsin(np.sqrt((np.arange(1, 6) - 0.5) / 5)) - np.arcsin(np.sqrt(u)))
        )
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Coles must not use MLE")

    monkeypatch.setattr(w.weibull_statistics, "_fitted", forbidden)
    monkeypatch.setattr(w.weibull_statistics, "_fit_logs", forbidden)
    test = criterion()
    assert abs(test.execute_statistic(sample) - old) > 1e-3
    for data in (sample, [0.1, 0.2, 0.3, 0.4, 10], sample):
        assert test.execute_statistic(data) == pytest.approx(reference(data))
    with pytest.raises(ValueError, match="regenerate calibration"):
        test._validate_storage_calibration()


def test_two_observations_are_degenerate():
    assert criterion().execute_statistic([1, 2]) == pytest.approx(
        criterion().execute_statistic([0.001, 100])
    )


@pytest.mark.parametrize("sample", [[], [1], [1, 1], [0, 1], [-1, 2], [np.inf, 1], [np.nan, 1]])
def test_invalid_samples(sample):
    with pytest.raises(ValueError):
        criterion().execute_statistic(sample)
