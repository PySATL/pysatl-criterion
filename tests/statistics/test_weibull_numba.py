"""Independent references for compiled Weibull transform and record kernels."""

import numpy as np
import pytest
from scipy.special import gamma, gammaln, logsumexp
from scipy.stats import gumbel_l

from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous import weibull as w


@pytest.mark.parametrize("size", [3, 20, 200])
@pytest.mark.parametrize("kind", [2, 3])
@pytest.mark.parametrize("a", [-5.0, 0.0, 10.0])
def test_laplace_matches_independent_fit_and_vectorized_reference(size, kind, a):
    sample = np.random.default_rng(size).weibull(1.7, size)
    location, scale = gumbel_l.fit(np.log(sample))
    z = (np.log(sample) - location) / scale
    m = 17
    t = (
        np.arange(-m, 0) / m
        if kind == 2
        else np.arange(int(np.ceil(-2.5 * m)), int(np.floor(0.49 * m)) + 1) / m
    )
    empirical = np.mean(np.exp(-np.outer(z, t)), axis=0)
    expected = size * np.sum(np.exp(a * t - np.exp(a * t)) * (empirical - gamma(1 - t)) ** 2)
    cls = getattr(w, f"LaplaceTransform{kind}WeibullGofStatistic")
    actual = cls(Weibull.DEFAULT.parse({})).execute_statistic(sample, m=m, a=a)
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10)
    assert cls.do_execute_statistic.nopython_signatures


@pytest.mark.parametrize("a", [-1e308, 1e308])
def test_laplace_extreme_weights(a):
    z = np.array([-1000.0, -2.0, 0.5])
    t = np.array([-2.5, -0.5, 0.0, 0.49])
    theoretical = gammaln(1 - t)
    empirical = logsumexp(-np.outer(z, t), axis=0) - np.log(len(z))
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        difference = np.maximum(empirical, theoretical) + np.log(
            -np.expm1(-np.abs(empirical - theoretical))
        )
        at = a * t
        weight = np.full_like(t, -np.inf)
        finite = np.isfinite(at)
        weight[finite] = at[finite] - np.exp(at[finite])
        expected = len(z) * np.sum(np.exp(weight + 2 * difference))
    actual = w.LaplaceTransformWeibullGofStatistic.do_execute_statistic(z, t, theoretical, a)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("size", [2, 30, 200])
def test_cabana_quiroz_matches_matrix_reference(size):
    sample = np.random.default_rng(size).weibull(0.8, size)
    location, scale = gumbel_l.fit(np.log(sample))
    z = (np.log(sample) - location) / scale
    s = np.array([-0.1, 0.02])
    v = np.sqrt(size) * (np.mean(np.exp(-np.outer(z, s)), axis=0) - gamma(1 - s))
    expected = v @ np.linalg.solve(np.array([[1.59, 0.91], [0.91, 0.53]]), v)
    cls = w.CabanaQuirozWeibullGofStatistic
    assert cls(Weibull.DEFAULT.parse({})).execute_statistic(sample) == pytest.approx(
        expected, rel=1e-8, abs=1e-10
    )
    assert cls.do_execute_statistic.nopython_signatures


def test_record_kernel_matches_interval_integral():
    from scipy.integrate import quad

    u = np.array([0.1, 0.35, 0.8])
    counts = np.array([3.0, 2.0, 1.0])
    risk = np.cumsum(counts[::-1])[::-1]
    survival = np.r_[1.0, np.cumprod(1 - 1 / risk)]
    edges = np.r_[0.0, u, 1.0]
    expected = (
        sum(
            quad(lambda f, s=s: (s - (1 - f)) ** 2 / f, lo, hi)[0]
            for s, lo, hi in zip(survival, edges[:-1], edges[1:], strict=True)
        )
        * counts.sum()
    )
    cls = w.MahdiDoostparastWeibullGofStatistic
    assert cls.do_execute_statistic(u, np.log(u), counts) == pytest.approx(expected)
    assert cls.do_execute_statistic.nopython_signatures
