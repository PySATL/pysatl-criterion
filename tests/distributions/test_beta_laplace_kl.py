"""Independent integral/entropy references for the two fitted Beta criteria."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln, hyp1f1
from scipy.stats import beta as scipy_beta
from scipy.stats import differential_entropy

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.hypothesis_testing.beta_bootstrap import parametric_bootstrap_beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    KullbackLeiblerBetaGofStatistic,
    RiadAbdElMaboodBetaGofStatistic,
)


CLASSES = [RiadAbdElMaboodBetaGofStatistic, KullbackLeiblerBetaGofStatistic]
SAMPLE = np.array([0.06, 0.13, 0.18, 0.27, 0.39, 0.45, 0.61, 0.79, 0.88])


@pytest.mark.parametrize("shapes", [(0.4, 0.7), (1, 1), (2, 5), (8, 2), (30, 40)])
@pytest.mark.parametrize("weight", [0.01, 0.5, 1, 10, 100])
def test_laplace_against_direct_integral(shapes, weight):
    x = np.random.default_rng(671).beta(*shapes, size=25)
    a, b, _, _ = scipy_beta.fit(x, floc=0, fscale=1)

    def integrand(u):
        t = u / weight
        # Derivatives of the empirical transform, independent of pairwise kernel.
        e = np.exp(-t * x)
        residual = t * np.mean(x**2 * e) - (a + b + t) * np.mean(x * e) + a * np.mean(e)
        return len(x) / weight * (residual / a) ** 2 * np.exp(-u)

    expected, error = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-9, limit=300)
    statistic = RiadAbdElMaboodBetaGofStatistic(Beta.DEFAULT.parse({}), weight=weight)
    assert statistic.execute_statistic(x) == pytest.approx(expected, rel=1e-7, abs=2e-11)
    assert error < 1e-8


@pytest.mark.parametrize("a,b", [(0.4, 0.7), (1, 1), (2, 5), (30, 40)])
@pytest.mark.parametrize("t", [0, 0.1, 1, 20])
def test_population_kummer_equation(a, b, t):
    transform = hyp1f1(a, a + b, -t)
    first = -a / (a + b) * hyp1f1(a + 1, a + b + 1, -t)
    second = a * (a + 1) / ((a + b) * (a + b + 1)) * hyp1f1(a + 2, a + b + 2, -t)
    assert t * second + (a + b + t) * first + a * transform == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("m", [1, 2, 4, None])
def test_kl_against_scipy_entropy_and_explicit_log_likelihood(m):
    a, b, _, _ = scipy_beta.fit(SAMPLE, floc=0, fscale=1)
    window = min(int(np.sqrt(len(SAMPLE))), (len(SAMPLE) - 1) // 2) if m is None else m
    entropy = differential_entropy(SAMPLE, window_length=window, method="vasicek")
    expected = -entropy + betaln(a, b) - (a - 1) * np.log(SAMPLE).mean()
    expected -= (b - 1) * np.log1p(-SAMPLE).mean()
    test = KullbackLeiblerBetaGofStatistic(Beta.DEFAULT.parse({}), window=m)
    assert test.execute_statistic(SAMPLE) == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("cls", CLASSES)
def test_contract_and_refitting(cls):
    for parameters in ({"a": 2}, {"b": 3}, {"a": 2, "b": 3}):
        with pytest.raises(ValueError):
            cls(Beta.DEFAULT.parse(parameters))
    test = cls(Beta.DEFAULT.parse({}))
    original = SAMPLE.copy()
    value = test.execute_statistic(SAMPLE)
    assert test.execute_statistic(SAMPLE[::-1]) == pytest.approx(value)
    test.execute_statistic([0.12, 0.3, 0.4, 0.8])
    assert test.execute_statistic(SAMPLE) == value
    assert test.hypothesis().parameters() == {}
    np.testing.assert_array_equal(SAMPLE, original)
    with pytest.raises(ValueError, match="bootstrap"):
        test._validate_storage_calibration()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "x",
    [
        [],
        [0.2],
        [0.2, 0.2, 0.2],
        [0, 0.2, 0.5],
        [0.2, 0.5, 1],
        [-0.1, 0.2, 0.5],
        [0.2, np.nan, 0.5],
        [0.2, np.inf, 0.5],
        [[0.1, 0.2, 0.3]],
        [0.1, 0.2j, 0.3],
        np.ma.array([0.1, 0.2, 0.3], mask=[0, 1, 0]),
    ],
)
def test_invalid_samples(cls, x):
    with pytest.raises(ValueError):
        cls(Beta.DEFAULT.parse({})).execute_statistic(x)


@pytest.mark.parametrize("weight", [0, -1, np.inf, np.nan, True, np.bool_(False), "1", 1j])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        RiadAbdElMaboodBetaGofStatistic(Beta.DEFAULT.parse({}), weight=weight)


@pytest.mark.parametrize("m", [0, -1, 1.5, True, np.bool_(False), "1", np.inf])
def test_invalid_window(m):
    with pytest.raises(ValueError):
        KullbackLeiblerBetaGofStatistic(Beta.DEFAULT.parse({}), window=m)


def test_window_size_ties_and_configuration():
    test = KullbackLeiblerBetaGofStatistic(Beta.DEFAULT.parse({}), window=2)
    with pytest.raises(ValueError, match="window must"):
        test.execute_statistic([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        test.execute_statistic([0.1, 0.2])
    with pytest.raises(ValueError, match="spacings"):
        test.execute_statistic([0.2] * 5 + [0.7] * 4)
    assert np.isfinite(test.execute_statistic([0.1, 0.1, 0.3, 0.5, 0.8]))
    assert test.configuration()["window"] == 2
    assert (
        RiadAbdElMaboodBetaGofStatistic(Beta.DEFAULT.parse({}), weight=2).configuration()["weight"]
        == 2
    )


@pytest.mark.parametrize(
    "cls,options",
    [
        (RiadAbdElMaboodBetaGofStatistic, {"weight": 0.3}),
        (KullbackLeiblerBetaGofStatistic, {"window": 1}),
    ],
)
def test_bootstrap_preserves_settings_and_refits(cls, options, monkeypatch):
    test = cls(Beta.DEFAULT.parse({}), **options)
    original_fit = cls._fit
    fits = []

    def fit(x):
        result = original_fit(x)
        fits.append(result[1:])
        return result

    monkeypatch.setattr(cls, "_fit", staticmethod(fit))
    actual = parametric_bootstrap_beta(test, SAMPLE, n_resamples=19, random_state=17)
    assert len(fits) == 21
    assert len(set(fits)) > 2
    x, a, b = original_fit(SAMPLE)
    rng = np.random.default_rng(17)
    reference = [test.execute_statistic(rng.beta(a, b, len(x))) for _ in range(19)]
    assert actual.p_value == (1 + np.count_nonzero(np.asarray(reference) >= actual.statistic)) / 20
    assert actual.critical_value == np.quantile(reference, 0.95, method="inverted_cdf")
    assert actual == parametric_bootstrap_beta(test, SAMPLE, n_resamples=19, random_state=17)
