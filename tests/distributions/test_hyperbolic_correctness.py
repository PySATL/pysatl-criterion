"""Regression checks for the fixed-parameter hyperbolic EDF criteria."""

import doctest
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k1

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit import hyperbolic


CLASSES = [
    hyperbolic.KolmogorovSmirnovHyperbolicGofStatistic,
    hyperbolic.CramerVonMisesHyperbolicGofStatistic,
    hyperbolic.AndersonDarlingHyperbolicGofStatistic,
    hyperbolic.KuiperHyperbolicGofStatistic,
    hyperbolic.WatsonHyperbolicGofStatistic,
]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("sample", [[], [[1, 2]], [np.nan], [np.inf], [-np.inf], [1j]])
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls().execute_statistic(sample)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("parameter", ["alpha", "beta", "delta", "mu"])
@pytest.mark.parametrize("value", [[1], np.array([1, 2]), 1j, "1", None, True])
def test_parameters_must_be_real_scalars(cls, parameter, value):
    with pytest.raises(ValueError):
        cls(**{parameter: value})


@pytest.mark.parametrize(
    "params", [{"alpha": 1e308, "delta": 2}, {"alpha": 1e-300, "delta": 1e-300}]
)
def test_unrepresentable_scaled_shape(params):
    with pytest.raises(ValueError, match="Scaled parameters"):
        CLASSES[1](**params)


@pytest.mark.parametrize("cls", CLASSES)
def test_independent_calls_and_fixed_hypothesis(cls):
    statistic = cls(alpha=2, beta=0.3, delta=1.5, mu=-0.5)
    sample = np.array([2.0, -1.0, 0.0, 0.0])
    original = sample.copy()
    first = statistic.execute_statistic(sample, unused=True)
    assert isinstance(first, float)
    assert isinstance(statistic.alternative(), RightAlternative)
    statistic.execute_statistic([3.0, 4.0])
    assert statistic.execute_statistic(sample) == first
    np.testing.assert_array_equal(sample, original)
    assert statistic.hypothesis().parameters() == {
        "alpha": 2,
        "beta": 0.3,
        "delta": 1.5,
        "mu": -0.5,
    }


@pytest.mark.parametrize("sample", [[0.0], [0.0, 0.0, 0.0], [-2.0, 0.2, 1.0, 3.0]])
def test_against_integrated_classical_density(sample):
    # Independent classical density, not SciPy's GH parameter conversion.
    alpha, beta, delta, mu = 2.0, 0.3, 1.5, -0.5
    gamma = np.sqrt(alpha**2 - beta**2)
    normalizer = gamma / (2 * alpha * delta * k1(delta * gamma))

    def density(x):
        y = x - mu
        return normalizer * np.exp(-alpha * np.hypot(delta, y) + beta * y)

    x = np.sort(sample)
    u = np.array([quad(density, -np.inf, value, epsabs=1e-12)[0] for value in x])
    sf = np.array([quad(density, value, np.inf, epsabs=1e-12)[0] for value in x])
    n = len(x)
    i = np.arange(1, n + 1)
    plus, minus = max(i / n - u), max(u - (i - 1) / n)
    w = 1 / (12 * n) + sum((u - (i - 0.5) / n) ** 2)
    expected = [
        max(plus, minus),
        w,
        -n - sum((2 * i - 1) / n * (np.log(u) + np.log(sf[::-1]))),
        plus + minus,
        w - n * (np.mean(u) - 0.5) ** 2,
    ]
    for cls, reference in zip(CLASSES, expected, strict=True):
        result = cls(alpha=alpha, beta=beta, delta=delta, mu=mu).execute_statistic(sample)
        assert result == pytest.approx(reference, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("direction", [AlternativeType.LEFT, AlternativeType.RIGHT])
def test_storage_rejects_ambiguous_ks_calibration(direction):
    store = Mock()
    resolver = StorageLimitDistributionResolver(store)
    with pytest.raises(ValueError, match="CDF direction"):
        resolver.resolve(CLASSES[0](alternative_type=direction), 10)
    store.get.assert_not_called()


def test_invalid_ks_direction():
    with pytest.raises(ValueError, match="AlternativeType"):
        CLASSES[0](alternative_type="right")


@pytest.mark.parametrize("cls", [CLASSES[i] for i in [0, 1, 3, 4]])
@pytest.mark.parametrize("values", [[np.nan, 0.5], [-0.1, 0.5], [0.5, 1.1], [0.9, 0.1]])
def test_cdf_numerical_failures_are_explicit(monkeypatch, cls, values):
    monkeypatch.setattr(hyperbolic.scipy_stats.genhyperbolic, "cdf", lambda *a, **k: values)
    with pytest.raises(FloatingPointError, match="CDF"):
        cls().execute_statistic([0.0, 1.0])


def test_ad_tail_underflow_is_not_mathematical_infinity(monkeypatch):
    monkeypatch.setattr(
        hyperbolic.scipy_stats.genhyperbolic, "logcdf", lambda *a, **k: np.array([-np.inf])
    )
    with pytest.raises(FloatingPointError, match="underflowed"):
        CLASSES[2]().execute_statistic([-1000.0])


def test_ad_does_not_subtract_cdf_from_one():
    # CDF rounds to one here, but direct survival integration is still nonzero.
    result = CLASSES[2]().execute_statistic([40.0])
    assert np.isfinite(result) and result > 30


def test_watson_centering_avoids_cancellation():
    n = 10000
    # Equally spaced probabilities translated within (0, 1): all d_i coincide.
    u = (np.arange(n) + 0.5) / n + 0.00004
    result = CLASSES[4]().do_execute_statistic(u)
    assert result == pytest.approx(1 / (12 * n), abs=1e-18)


def test_docstring_examples():
    assert doctest.testmod(hyperbolic).failed == 0
