"""Contracts and independent numerical checks for generated exponential criteria."""

import inspect
import pickle

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import ExponentialDistributionDescriptor as E
from pysatl_criterion.hypothesis_testing.exponential_monte_carlo import monte_carlo_exponential
from pysatl_criterion.statistics.goodness_of_fit import edf, frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    EqualProbabilityBins,
    ExplicitBins,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous import exponent
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponent.exponent_statistics import (
    _EDF_STATISTICS,
    _FREQUENCY_STATISTICS,
)


CATALOG = {**_EDF_STATISTICS, **_FREQUENCY_STATISTICS}
SAMPLE = np.array([0.0, 0.12, 0.3, 0.7, 1.2, 1.8, 2.5, 4.0])


def test_catalog_covers_shared_algorithms():
    for source, catalog, base in (
        (edf, _EDF_STATISTICS, edf.AbstractEDFStatistic),
        (frequency, _FREQUENCY_STATISTICS, frequency.AbstractFrequencyStatistic),
    ):
        algorithms = {
            cls
            for cls in vars(source).values()
            if inspect.isclass(cls) and issubclass(cls, base) and cls is not base
        }
        assert {spec.algorithm for spec in catalog.values()} == algorithms
    assert len({getattr(exponent, name).code() for name in CATALOG}) == len(CATALOG)


@pytest.mark.parametrize("name", CATALOG)
def test_generated_hypotheses_and_serialization(name):
    cls = getattr(exponent, name)
    assert name in exponent.__all__
    assert cls.__module__ == exponent.__name__
    assert "supported_hypotheses" in vars(cls)
    assert cls.supported_hypotheses() == CATALOG[name].supported_hypotheses
    assert cls.do_execute_statistic is CATALOG[name].algorithm.do_execute_statistic
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301 - local class
    for support in cls.supported_hypotheses():
        values = {"lam": 1.7} if support.fixed_parameters else {}
        obj = cls(E.DEFAULT.parse(values))
        assert not np.isnan(obj.execute_statistic(SAMPLE))
        assert obj.hypothesis().parameters() == values


@pytest.mark.parametrize("name", CATALOG)
def test_validation_and_nonmutation(name):
    cls = getattr(exponent, name)
    for support in cls.supported_hypotheses():
        obj = cls(E.DEFAULT.parse({"lam": 1} if support.fixed_parameters else {}))
        for bad in (
            [],
            [-1, 1],
            [[1, 2]],
            [1, np.inf],
            [1, np.nan],
            [1j, 2],
            np.ma.array([1, 2], mask=[1, 0]),
        ):
            with pytest.raises(ValueError):
                obj.execute_statistic(bad)
        if not support.fixed_parameters:
            for bad in ([1], [0, 0]):
                with pytest.raises(ValueError):
                    obj.execute_statistic(bad)
        sample = SAMPLE.copy()
        state = vars(obj).copy()
        value = obj.execute_statistic(sample)
        obj.execute_statistic(sample + 1)
        assert obj.execute_statistic(sample[::-1]) == pytest.approx(value)
        np.testing.assert_array_equal(sample, SAMPLE)
        assert vars(obj) == state


@pytest.mark.parametrize("name", CATALOG)
@pytest.mark.parametrize("scale", [1e-300, 7.3, 1e300])
def test_fitted_scale_invariance(name, scale):
    cls = getattr(exponent, name)
    if not any(not support.fixed_parameters for support in cls.supported_hypotheses()):
        return
    obj = cls(E.DEFAULT.parse({}))
    assert obj.execute_statistic(SAMPLE * scale) == pytest.approx(obj.execute_statistic(SAMPLE))


@pytest.mark.parametrize("rate", [None, 1.7])
def test_cvm_and_lilliefors_scipy_oracles(rate):
    values = {} if rate is None else {"lam": rate}
    scale = SAMPLE.mean() if rate is None else 1 / rate
    cdf = stats.expon(scale=scale).cdf
    cvm = exponent.CramerVonMisesExponentialityGofStatistic(E.DEFAULT.parse(values))
    assert cvm.execute_statistic(SAMPLE) == pytest.approx(
        stats.cramervonmises(SAMPLE, cdf).statistic
    )
    if rate is None:
        ks = exponent.LillieforsExponentialityGofStatistic(E.DEFAULT.parse({}))
        assert ks.execute_statistic(SAMPLE) == pytest.approx(stats.kstest(SAMPLE, cdf).statistic)


@pytest.mark.parametrize("rate", [None, 1.7])
@pytest.mark.parametrize(
    "name,power",
    [
        ("Chi2Pearson", 1),
        ("FreemanTukey", -0.5),
        ("ModifiedLogLikelihood", -1),
        ("NeymanChi2", -2),
        ("CressieRead", 2 / 3),
    ],
)
def test_frequency_scipy_oracle(name, power, rate):
    sample = np.linspace(0.02, 4, 80)
    scale = sample.mean() if rate is None else 1 / rate
    edges = stats.expon.ppf(np.linspace(0, 1, 5), scale=scale)
    observed = np.histogram(sample, edges)[0]
    expected = stats.power_divergence(observed, np.full(4, 20), lambda_=power).statistic
    cls = getattr(exponent, name + "ExponentialityGofStatistic")
    obj = cls(
        E.DEFAULT.parse({} if rate is None else {"lam": rate}),
        binning=EqualProbabilityBins(count=4),
    )
    assert obj.execute_statistic(sample) == pytest.approx(expected)


def test_binning_calibration_guard():
    cls = exponent.Chi2PearsonExponentialityGofStatistic
    bins = ExplicitBins([0, 1, 2, np.inf])
    fitted = cls(E.DEFAULT.parse({}), binning=bins)
    assert np.isfinite(fitted.execute_statistic(SAMPLE))
    with pytest.raises(ValueError, match="external calibration"):
        monte_carlo_exponential(fitted, SAMPLE, n_resamples=5)
    fixed = cls(E.DEFAULT.parse({"lam": 1}), binning=bins)
    result = monte_carlo_exponential(fixed, SAMPLE, n_resamples=5, random_state=1)
    assert 0 < result.p_value <= 1
