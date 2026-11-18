"""Analytical MLE references, likelihood equations, and invalid sample contracts."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import special, stats

from pysatl_criterion.estimation.base import AbstractParameterEstimator, EstimationMethod
from pysatl_criterion.estimation.maximum_likelihood import estimators as mle


ESTIMATORS = [
    cls
    for _, cls in inspect.getmembers(mle, inspect.isclass)
    if issubclass(cls, AbstractParameterEstimator) and not inspect.isabstract(cls)
]


@pytest.fixture(params=ESTIMATORS, ids=lambda cls: cls.__name__)
def estimator(request):
    return request.param(n=4) if request.param is mle.BinomialMleEstimator else request.param()


@pytest.mark.parametrize(
    "sample",
    [
        [],
        1,
        [[1, 2]],
        [np.nan, 1],
        [np.inf, 1],
        [1j, 1],
        ["bad", 1],
        np.ma.array([1, 2], mask=[False, True]),
    ],
)
def test_all_mle_estimators_reject_invalid_samples(estimator, sample):
    with pytest.raises(ValueError, match="Sample"):
        estimator.estimate(sample)


def test_all_estimators_declare_mle_method(estimator):
    assert estimator.method() is EstimationMethod.MLE


@pytest.mark.parametrize(
    "cls,sample,expected",
    [
        (mle.UniformMleEstimator, [-2, 1, 4], {"a": -2, "b": 4}),
        (mle.DiscreteUniformMleEstimator, [-2, 1, 4], {"a": -2, "b": 4}),
        (mle.NormalMleEstimator, [-2, 1, 4], {"mean": 1, "var": 6}),
        (mle.LogNormalMleEstimator, np.exp([-2, 1, 4]), {"mean": 1, "var": 6}),
        (mle.ExponentialMleEstimator, [0, 1, 2, 5], {"lam": 0.5}),
        (mle.ParetoMleEstimator, [2, 2 * np.e, 2 * np.e**2], {"x_m": 2, "alpha": 1}),
        (mle.LaplaceMleEstimator, [-2, 1, 7], {"t": 1, "s": 3}),
        (mle.RayleighMleEstimator, [0, 2, 4], {"scale": np.sqrt(10 / 3)}),
        (mle.BernoulliMleEstimator, [0, 1, 1, 1], {"p": 0.75}),
        (mle.PoissonMleEstimator, [0, 1, 2, 5], {"lam": 2}),
        (mle.GeometricMleEstimator, [1, 2, 3], {"p": 0.5}),
    ],
)
def test_closed_form_estimates(cls, sample, expected):
    original = np.array(sample, dtype=float)
    saved = original.copy()
    original.flags.writeable = False
    assert cls().estimate(original) == pytest.approx(expected)
    assert cls().estimate(original[::-1]) == pytest.approx(expected)
    np.testing.assert_array_equal(original, saved)


def test_binomial_uses_number_of_trials():
    assert mle.BinomialMleEstimator(4).estimate([0, 1, 2, 3]) == {"p": 0.375}
    with pytest.raises(ValueError, match="positive"):
        mle.BinomialMleEstimator(0)


@pytest.mark.parametrize(
    "cls,sample,message",
    [
        (mle.LogNormalMleEstimator, [0, 1], "strictly positive"),
        (mle.ExponentialMleEstimator, [-1, 1], "non-negative"),
        (mle.WeibullMleEstimator, [0, 1], "strictly positive"),
        (mle.GammaMleEstimator, [-1, 1], "strictly positive"),
        (mle.BetaMleEstimator, [0, 0.5], "strictly inside"),
        (mle.BetaMleEstimator, [0.5, 1], "strictly inside"),
        (mle.CauchyMleEstimator, [2, 2], "non-degenerate"),
        (mle.ChiSquaredMleEstimator, [0, 1], "strictly positive"),
        (mle.StudentMleEstimator, [2, 2], "non-degenerate"),
        (mle.FisherMleEstimator, [-1, 1], "strictly positive"),
        (mle.FisherMleEstimator, [2, 2], "non-degenerate"),
        (mle.RayleighMleEstimator, [-1, 1], "non-negative"),
        (mle.WignerMleEstimator, [2, 2], "non-degenerate"),
        (mle.ParetoMleEstimator, [0, 1], "strictly positive"),
        (mle.ParetoMleEstimator, [2, 2], "undefined"),
        (mle.LaplaceMleEstimator, [2, 2], "non-degenerate"),
        (mle.BernoulliMleEstimator, [0, 2], "binary"),
        (mle.PoissonMleEstimator, [-1, 1], "non-negative integer"),
        (mle.PoissonMleEstimator, [0.5, 1], "non-negative integer"),
        (mle.GeometricMleEstimator, [0, 1], "positive integer"),
        (mle.GeometricMleEstimator, [1.5, 2], "positive integer"),
    ],
)
def test_support_and_degeneracy_errors(cls, sample, message):
    with pytest.raises(ValueError, match=message):
        cls().estimate(sample)


@pytest.mark.parametrize("sample", [[-1, 2], [1, 5], [0.5, 1]])
def test_binomial_rejects_invalid_counts(sample):
    with pytest.raises(ValueError, match="integer counts"):
        mle.BinomialMleEstimator(4).estimate(sample)


def test_gamma_satisfies_likelihood_equations():
    x = np.array([0.2, 0.6, 1, 1.5, 3, 7])
    result = mle.GammaMleEstimator().estimate(x)
    shape, rate = result["alfa"], result["beta"]
    assert shape / rate == pytest.approx(x.mean())
    assert special.digamma(shape) - np.log(rate) == pytest.approx(np.log(x).mean(), abs=1e-8)


def test_weibull_satisfies_likelihood_equations():
    x = np.array([0.2, 0.6, 1, 1.5, 3, 7])
    result = mle.WeibullMleEstimator().estimate(x)
    scale, shape = result["a"], result["k"]
    weights = (x / scale) ** shape
    assert weights.mean() == pytest.approx(1, abs=1e-4)
    assert 1 / shape + np.log(x).mean() == pytest.approx(np.mean(weights * np.log(x)), abs=1e-4)


def test_beta_satisfies_both_likelihood_equations():
    x = np.array([0.05, 0.12, 0.2, 0.3, 0.45, 0.7])
    result = mle.BetaMleEstimator().estimate(x)
    a, b = result["a"], result["b"]
    assert special.digamma(a) - special.digamma(a + b) == pytest.approx(np.log(x).mean())
    assert special.digamma(b) - special.digamma(a + b) == pytest.approx(np.log1p(-x).mean())


def test_chi_squared_estimates_df_without_fitting_a_hidden_scale():
    x = np.array([0.2, 0.6, 1, 1.5, 3, 7])
    df = mle.ChiSquaredMleEstimator().estimate(x)["df"]
    assert special.digamma(df / 2) == pytest.approx(np.log(x).mean() - np.log(2), abs=1e-4)


@pytest.mark.parametrize(
    "cls,law,keys",
    [
        (mle.CauchyMleEstimator, stats.cauchy, ("t", "s")),
        (mle.StudentMleEstimator, stats.t, ("df", "loc", "scale")),
        (mle.FisherMleEstimator, stats.f, ("dfn", "dfd", "loc", "scale")),
    ],
)
def test_numerical_fit_is_a_local_likelihood_maximum(cls, law, keys):
    x = np.array([0.1, 0.2, 0.25, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5, 12])
    result = cls().estimate(x)
    parameters = np.array([result[key] for key in keys])
    fitted = law.logpdf(x, *parameters).sum()
    assert np.isfinite(fitted)
    for i, key in enumerate(keys):
        if cls is mle.FisherMleEstimator and key == "loc":
            assert result[key] == 0  # Fixed by this estimator's contract.
            continue
        for direction in (-1, 1):
            perturbed = parameters.copy()
            perturbed[i] += direction * 0.01 * max(abs(parameters[i]), 1)
            assert law.logpdf(x, *perturbed).sum() <= fitted + 1e-7


def test_wigner_radius_solves_likelihood_equation():
    x = np.array([-2, -0.5, 0, 1, 2.5])
    radius = mle.WignerMleEstimator().estimate(x)["R"]
    assert radius > max(abs(x))
    assert radius**2 * np.sum(1 / (radius**2 - x**2)) == pytest.approx(2 * len(x), rel=1e-5)


def test_wigner_optimizer_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        mle, "minimize_scalar", lambda *args, **kwargs: SimpleNamespace(success=False)
    )
    with pytest.raises(ValueError, match="Failed to estimate"):
        mle.WignerMleEstimator().estimate([-1, 0, 1])
