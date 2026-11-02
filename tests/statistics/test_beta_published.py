"""Independent references for the published Laplace and Vasicek-KL Beta tests."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta, differential_entropy

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.hypothesis_testing.beta_bootstrap import parametric_bootstrap_beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    KullbackLeiblerBetaGofStatistic as KL,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    RiadAbdElMaboodBetaGofStatistic as RF,
)


@pytest.mark.parametrize("shapes", [(0.4, 0.7), (2, 5), (7, 2), (30, 40)])
@pytest.mark.parametrize("weight", [0.01, 0.25, 2, 5, 100])
@pytest.mark.parametrize("n", [7, 40])
def test_laplace_against_ode_integral(shapes, weight, n):
    x = np.random.default_rng(123).beta(*shapes, size=n)
    a, b, _, _ = beta.fit(x, floc=0, fscale=1)

    def integrand(t):
        exp = np.exp(-t * x)
        residual = (t * np.mean(x * x * exp) - (a + b + t) * np.mean(x * exp)) / a + np.mean(exp)
        return n * residual**2 * np.exp(-weight * t)

    expected = quad(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-10)[0]
    actual = RF(Beta.DEFAULT.parse({}), weight=weight).execute_statistic(x)
    assert actual == pytest.approx(expected, rel=2e-7, abs=2e-12)
    assert actual >= 0


@pytest.mark.parametrize("window", [1, 2, 4, None])
@pytest.mark.parametrize("shapes", [(0.4, 0.7), (2, 5), (30, 40)])
def test_kl_against_scipy_vasicek(shapes, window):
    x = np.random.default_rng(23).beta(*shapes, size=15)
    a, b, _, _ = beta.fit(x, floc=0, fscale=1)
    m = window if window is not None else 3
    expected = -differential_entropy(x, window_length=m, method="vasicek") - np.mean(
        beta.logpdf(x, a, b)
    )
    criterion = KL(Beta.DEFAULT.parse({}), window=window)
    assert criterion.execute_statistic(x) == pytest.approx(expected, abs=1e-12)
    assert criterion.execute_statistic(1 - x) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("cls,settings", [(RF, {"weight": 0.25}), (KL, {"window": 2})])
def test_bootstrap_and_configuration(cls, settings):
    x = np.array([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    saved = x.copy()
    test = cls(Beta.DEFAULT.parse({}), **settings)
    for key, val in settings.items():
        assert test.configuration()[key] == val
    assert test.execute_statistic(x) == pytest.approx(test.execute_statistic(x[::-1]), abs=1e-12)
    observed = test.execute_statistic(x)
    a, b, _, _ = beta.fit(x, floc=0, fscale=1)
    rng = np.random.default_rng(21)
    values = [test.execute_statistic(rng.beta(a, b, len(x))) for _ in range(19)]
    result = parametric_bootstrap_beta(test, x, n_resamples=19, random_state=21)
    assert result.statistic == observed
    assert result.p_value == (1 + np.count_nonzero(np.asarray(values) >= observed)) / 20
    assert result.critical_value == np.quantile(values, 0.95, method="inverted_cdf")
    np.testing.assert_array_equal(saved, x)
    with pytest.raises(ValueError, match="Composite Beta"):
        test._validate_storage_calibration()
    with pytest.raises(ValueError):
        cls(Beta.DEFAULT.parse({"a": 2, "b": 3}), **settings)


@pytest.mark.parametrize("weight", [0, -1, np.nan, np.inf, True, "2", 1j])
def test_invalid_weight(weight):
    with pytest.raises(ValueError, match="weight"):
        RF(Beta.DEFAULT.parse({}), weight=weight)


@pytest.mark.parametrize("window", [0, -1, 1.5, np.nan, True, "2"])
def test_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        KL(Beta.DEFAULT.parse({}), window=window)


@pytest.mark.parametrize("cls", [RF, KL])
@pytest.mark.parametrize(
    "sample",
    [
        [],
        [0.5],
        [0.3, 0.3, 0.3],
        [0, 0.2, 0.4],
        [0.1, 0.2, 1],
        [0.1, np.nan, 0.3],
        [[0.1, 0.2, 0.3]],
        [0.1, 0.2, 1j],
    ],
)
def test_invalid_sample(cls, sample):
    with pytest.raises(ValueError):
        cls(Beta.DEFAULT.parse({})).execute_statistic(sample)


def test_window_boundaries_and_zero_spacings():
    with pytest.raises(ValueError, match="window"):
        KL(Beta.DEFAULT.parse({}), window=2).execute_statistic([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="spacings"):
        KL(Beta.DEFAULT.parse({}), window=1).execute_statistic([0.1, 0.1, 0.1, 0.4, 0.5])
    assert np.isfinite(
        KL(Beta.DEFAULT.parse({}), window=2).execute_statistic([0.1, 0.1, 0.3, 0.4, 0.5])
    )
    assert np.isfinite(KL(Beta.DEFAULT.parse({})).execute_statistic([0.1, 0.2, 0.3]))


def test_laplace_cancellation_fallback(monkeypatch):
    test = RF(Beta.DEFAULT.parse({}))
    x = [0.1, 0.2, 0.3, 0.5]
    expected = test.execute_statistic(x)
    monkeypatch.setattr(test, "_laplace_kernel", lambda *args: (-1e-16, 1.0))
    assert test.execute_statistic(x) == pytest.approx(expected, rel=1e-9)


def test_negative_kl_is_not_clipped():
    x = np.array([1e-12, 0.01, 0.1, 0.3, 0.7, 0.9, 0.99, 1 - 1e-12])
    test = KL(Beta.DEFAULT.parse({}), window=2)
    a, b, _, _ = beta.fit(x, floc=0, fscale=1)
    expected = -differential_entropy(x, window_length=2, method="vasicek") - np.mean(
        beta.logpdf(x, a, b)
    )
    assert expected < 0
    assert test.execute_statistic(x) == pytest.approx(expected)
