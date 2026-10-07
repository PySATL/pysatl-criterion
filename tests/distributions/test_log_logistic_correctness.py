import doctest
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import CensoredData, fisk, logistic

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import log_logistic as module
from pysatl_criterion.statistics.goodness_of_fit.log_logistic import (
    AndersonDarlingLogLogisticGofStatistic as AD,
)
from pysatl_criterion.statistics.goodness_of_fit.log_logistic import (
    Chi2PearsonLogLogisticGofStatistic as Pearson,
)
from pysatl_criterion.statistics.goodness_of_fit.log_logistic import (
    CramerVonMisesLogLogisticGofStatistic as CVM,
)
from pysatl_criterion.statistics.goodness_of_fit.log_logistic import (
    KolmogorovSmirnovLogLogisticGofStatistic as KS,
)
from pysatl_criterion.statistics.goodness_of_fit.log_logistic import (
    MirvalievLogLogisticGofStatistic as Mirvaliev,
)
from pysatl_criterion.statistics.goodness_of_fit.log_logistic import (
    NikulinLogLogisticGofStatistic as Nikulin,
)


@pytest.mark.parametrize("cls", [KS, AD, CVM, Pearson, Mirvaliev, Nikulin])
@pytest.mark.parametrize("sample", [[], [[1, 2]], [0, 1], [-1, 2], [np.inf], [np.nan], [1j]])
def test_invalid_sample(cls, sample):
    with pytest.raises(ValueError):
        cls().execute_statistic(sample, unused=True)


@pytest.mark.parametrize("cls", [KS, AD, CVM, Pearson])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, [1], 1j, True])
@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_invalid_parameters(cls, value, name):
    with pytest.raises(ValueError):
        cls(**{name: value})


@pytest.mark.parametrize(
    "cls,key", [(Pearson, "bins"), (Nikulin, "n_intervals"), (Mirvaliev, "n_intervals")]
)
@pytest.mark.parametrize("value", [1, 2.5, True, np.nan, np.inf])
def test_interval_validation(cls, key, value):
    with pytest.raises(ValueError):
        cls(**{key: value})


@pytest.mark.parametrize("direction", list(AlternativeType))
def test_ks_formula_and_tail(direction):
    x = np.array([0.1, 0.5, 2, 4])
    u = fisk.cdf(x, 2, scale=3)
    plus = max(np.arange(1, 5) / 4 - u)
    minus = max(u - np.arange(4) / 4)
    expected = {
        AlternativeType.RIGHT: plus,
        AlternativeType.LEFT: minus,
        AlternativeType.TWO_TAILED: max(plus, minus),
    }[direction]
    stat = KS(direction, alpha=3, beta=2)
    assert stat.execute_statistic(x) == pytest.approx(expected)
    assert stat.alternative().type() == AlternativeType.RIGHT


def test_extreme_ad_retains_finite_logs():
    x = [1e-250, 1e250]
    z = np.log(x) * 2
    expected = -2 - np.dot([0.5, 1.5], -np.logaddexp(0, -z) - np.logaddexp(0, z[::-1]))
    assert AD(beta=2).execute_statistic(x) == pytest.approx(expected)
    assert np.isfinite(AD(alpha=1e-250, beta=0.1).execute_statistic([1e250]))


def test_pearson_extreme_parameters_and_empty_bins():
    assert Pearson(bins=4, alpha=1e-250, beta=0.001).execute_statistic([1e250]) == 3


def test_mirvaliev_independent_covariance_formula():
    x = np.exp(np.random.default_rng(83).logistic(size=250))
    r = 6
    y = np.log(x)
    y = (y - y.mean()) / y.std()
    s = np.sqrt(3) / np.pi
    p = np.linspace(0, 1, r + 1)
    edges = np.r_[-np.inf, s * logit(p[1:-1]), np.inf]

    # Integrate the influence of fitted moments on each cell directly.
    def density(t):
        return logistic.pdf(t, scale=s)

    def cdf_derivatives(t):
        if not np.isfinite(t):
            return np.zeros(2)
        return -density(t) * np.array([1, t])

    b = np.array(
        [cdf_derivatives(hi) - cdf_derivatives(lo) for lo, hi in pairwise(edges)]
    ) * np.sqrt(r)
    c = np.array(
        [
            [
                quad(lambda t, k=k: (t if k == 0 else (t * t - 1) / 2) * density(t), lo, hi)[0]
                for k in range(2)
            ]
            for lo, hi in pairwise(edges)
        ]
    ) * np.sqrt(r)
    covariance = np.eye(r) - np.ones((r, r)) / r - b @ c.T - c @ b.T + b @ np.diag([1, 4 / 5]) @ b.T
    counts = np.histogram(y, edges)[0]
    residual = (counts - len(y) / r) / np.sqrt(len(y) / r)
    expected = residual @ np.linalg.pinv(covariance) @ residual
    assert Mirvaliev(r).execute_statistic(x) == pytest.approx(expected, rel=1e-8)
    assert abs(expected - residual @ residual) > 0.01


def test_nikulin_independent_likelihood_and_risk_integral():
    rng = np.random.default_rng(418)
    lifetime = np.exp(rng.logistic(size=600))
    times = np.minimum(lifetime, 4)
    events = lifetime <= 4
    # Independent SciPy fit to the censored logistic likelihood.
    mu, scale = logistic.fit(CensoredData.right_censored(np.log(times), ~events))

    def h(t):
        return expit((np.log(t) - mu) / scale) / (scale * t)

    def cumulative(t):
        return np.logaddexp(0, (np.log(t) - mu) / scale) if t else 0

    total = sum(cumulative(t) for t in times)
    r = 3
    bounds = (
        [0]
        + [
            brentq(
                lambda a, j=j: sum(cumulative(min(t, a)) for t in times) - j * total / r,
                1e-20,
                max(times),
            )
            for j in range(1, r)
        ]
        + [max(times)]
    )
    expected = []
    observed = []
    scores = []
    for lo, hi in pairwise(bounds):
        cuts = sorted(set([lo, hi] + list(times[(times > lo) & (times < hi)])))
        exposure = sum(np.count_nonzero(times >= b) * quad(h, a, b)[0] for a, b in pairwise(cuts))
        expected.append(exposure)
        event_times = times[(times > lo) & (times <= hi) & events]
        observed.append(len(event_times))
        z = (np.log(event_times) - mu) / scale
        scores.append(np.column_stack((-expit(-z), -1 - z * expit(-z))))
    n = len(times)
    a = np.diag(np.array(observed) / n)
    c = np.array([v.sum(axis=0) for v in scores]).T / n
    information = sum(v.T @ v for v in scores) / n
    covariance = a - c.T @ np.linalg.solve(information, c)
    residual = (np.array(observed) - expected) / np.sqrt(n)
    reference = residual @ np.linalg.solve(covariance, residual)
    assert Nikulin(r).execute_statistic((times, events)) == pytest.approx(reference, rel=2e-3)


@pytest.mark.parametrize("cls", [Mirvaliev, Nikulin])
def test_independence_invariance_and_no_mutation(cls):
    x = np.exp(np.random.default_rng(41).logistic(size=250))
    saved = x.copy()
    stat = cls(n_intervals=3)
    first = stat.execute_statistic(x, unused=True)
    assert isinstance(first, float)
    assert stat.hypothesis().parameters() == {}
    assert stat.alternative().type() == AlternativeType.RIGHT
    assert stat.execute_statistic(10 * x**2) == pytest.approx(first, rel=1e-7)
    stat.execute_statistic(np.exp(np.linspace(-2, 2, 100)))
    assert stat.execute_statistic(x) == pytest.approx(first)
    np.testing.assert_array_equal(x, saved)


@pytest.mark.parametrize("cls", [Mirvaliev, Nikulin])
@pytest.mark.parametrize(
    "data",
    [
        ([1, 2, 3], [1, 0]),
        ([1, 2, 3], [1, 2, 0]),
        ([1, 1, 1], [1, 1, 1]),
        ([1, 2, 3], [0, 0, 0]),
        ([1, 2, 3], [1, np.nan, 0]),
    ],
)
def test_fitted_invalid_data(cls, data):
    with pytest.raises(ValueError):
        cls().execute_statistic(data)


def test_empty_event_cells_are_infinite():
    assert np.isinf(Nikulin(10).execute_statistic([1, 2, 3]))


def test_calibration_guards(mocker):
    storage = StorageLimitDistributionResolver(mocker.Mock())
    for stat in [Nikulin(), Mirvaliev(), Pearson(), KS(AlternativeType.LEFT)]:
        with pytest.raises(ValueError):
            storage.resolve(stat, 100)
    with pytest.raises(ValueError, match="censoring plan"):
        MonteCarloLimitDistributionResolver(2).resolve(Nikulin(), 100)
    for stat in [Mirvaliev(3), KS(), AD(), CVM(), Pearson()]:
        with pytest.raises(ValueError, match="external calibration"):
            MonteCarloLimitDistributionResolver(2).resolve(stat, 100)


def test_examples():
    assert doctest.testmod(module).failed == 0
