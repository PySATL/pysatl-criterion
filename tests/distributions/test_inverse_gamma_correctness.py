"""Independent mathematical and interface regressions for Inverse Gamma tests."""

import doctest
import inspect
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution.distributions import InverseGammaDistributionDescriptor
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import inverse_gamma as module
from tests.parameter_cases import parameters_for


CLASSES = [
    cls
    for name, cls in vars(module).items()
    if name.endswith("InverseGammaGofStatistic") and not inspect.isabstract(cls)
]
FIXED = [cls for cls in CLASSES if cls is not module.LillieforsInverseGammaGofStatistic]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "sample",
    [
        [],
        [[1.0, 2.0]],
        [np.nan],
        [np.inf],
        [0.0, 1.0],
        [-1.0, 2.0],
        [1 + 1j],
        ["1", "2"],
        [True, False],
    ],
)
def test_sample_validation(cls, sample):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(sample, compatibility=True)


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, [1.0], 1j, True, "1"])
@pytest.mark.parametrize("parameter", ["alpha", "beta"])
def test_parameter_validation(cls, parameter, value):
    with pytest.raises(ValueError):
        cls(parameters_for(cls, **{parameter: value}))


@pytest.mark.parametrize("cls", CLASSES)
def test_interface_and_independence(cls):
    statistic = cls(parameters_for(cls))
    x = np.array([2.0, 0.3, 0.7, 1.1])
    original = x.copy()
    first = statistic.execute_statistic(x, compatibility=True)
    assert isinstance(first, (float, np.float64))
    assert np.isfinite(first)
    statistic.execute_statistic([0.6, 0.9, 3.0, 4.0])
    assert statistic.execute_statistic(x) == first
    np.testing.assert_array_equal(x, original)
    assert statistic.alternative().type() == AlternativeType.RIGHT


@pytest.mark.parametrize("cls", FIXED)
def test_fixed_parameter_scaling_and_hypothesis(cls):
    x = np.array([0.4, 0.8, 1.3, 2.7])
    statistic = cls(parameters_for(cls, alpha=3, beta=2))
    assert statistic.hypothesis().parameters() == {"alpha": 3, "beta": 2}
    assert statistic.execute_statistic(x) == pytest.approx(
        cls(parameters_for(cls, alpha=3, beta=14)).execute_statistic(x * 7), rel=2e-13
    )
    assert np.isfinite(statistic.execute_statistic([1.0]))
    assert np.isfinite(statistic.execute_statistic([1.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    "direction, scipy_direction",
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_directions(direction, scipy_direction):
    x = np.array([0.4, 0.8, 1.3, 2.7])
    expected = stats.kstest(
        x, stats.invgamma(3, scale=2).cdf, alternative=scipy_direction
    ).statistic
    assert module.KolmogorovSmirnovInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3, "beta": 2}),
        alternative_type=direction,
    ).execute_statistic(x) == pytest.approx(expected)


@pytest.mark.parametrize("bins", [True, 1, 2.5, np.nan, np.inf, [3]])
def test_bins_validation(bins):
    with pytest.raises(ValueError):
        module.Chi2PearsonInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1}), bins=bins
        )


@pytest.mark.parametrize("beta", [0.25, 2.0, 7.0])
def test_balanced_quantile_bins(beta):
    # Generate via reciprocal Gamma quantiles, independently of invgamma.ppf.
    x = beta / stats.gamma.ppf((np.arange(40) + 0.5) / 40, a=3)
    assert (
        module.Chi2PearsonInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3, "beta": beta}), bins=8
        ).execute_statistic(x)
        == 0
    )


def test_pearson_zero_counts_and_edges():
    statistic = module.Chi2PearsonInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 2}), bins=4
    )
    # For shape=1, F(x)=exp(-2/x). An interior quantile belongs to the right bin.
    edge = stats.invgamma.ppf(0.5, a=1, scale=2)
    counts, expected = statistic._counts_and_expected([edge, edge, 100.0])
    np.testing.assert_array_equal(counts, [0, 0, 2, 1])
    assert statistic.execute_statistic([edge, edge, 100.0]) == pytest.approx(
        sum((counts - expected) ** 2 / expected)
    )


def test_fitted_ks_moments_and_scale_invariance():
    statistic = module.LillieforsInverseGammaGofStatistic(
        InverseGammaDistributionDescriptor.DEFAULT.parse({})
    )
    assert statistic.hypothesis().parameters() == {}
    for x in [np.array([0.4, 0.7, 1.2, 2.0]), np.array([0.8, 1.0, 3.0, 9.0])]:
        mean = sum(x) / len(x)
        var = sum((x - mean) ** 2) / (len(x) - 1)
        a = 2 + mean**2 / var
        b = mean * (a - 1)
        expected = stats.kstest(x, lambda t, b=b, a=a: stats.gamma.sf(b / t, a=a)).statistic
        assert statistic.execute_statistic(x) == pytest.approx(expected)
        for scale in [1e-200, 1e200]:
            assert statistic.execute_statistic(x * scale) == pytest.approx(expected)
    with pytest.raises(ValueError, match="Unsupported"):
        module.LillieforsInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3})
        )
    for x in [[1.0], [1.0, 1.0]]:
        with pytest.raises(ValueError):
            statistic.execute_statistic(x)


@pytest.mark.parametrize(
    "cls",
    [
        module.AndersonDarlingInverseGammaGofStatistic,
        module.ZhangAInverseGammaGofStatistic,
        module.ZhangCInverseGammaGofStatistic,
        module.ZhangKInverseGammaGofStatistic,
        module.MinToshiyukiInverseGammaGofStatistic,
    ],
)
def test_numerical_underflow_is_explicit(cls):
    with pytest.raises(FloatingPointError):
        cls(parameters_for(cls)).execute_statistic([1e-300, 1.0])


@pytest.mark.parametrize("prefix", ["ZhangA", "ZhangC", "ZhangK"])
def test_zhang_unclipped_extreme_upper_tail(prefix):
    # Shape one has the exact CDF exp(-beta/x); logsf stays accurate when CDF rounds to 1.
    x = np.array([1.0, 1e20])
    log_u = -2 / x
    log_v = np.log(-np.expm1(log_u))
    i = np.array([1.0, 2.0])
    expected = {
        "ZhangA": -sum(log_u / (2 - i + 0.5) + log_v / (i - 0.5)),
        "ZhangC": sum((log_v - log_u - np.log((2 - i + 0.25) / (i - 0.75))) ** 2),
        "ZhangK": max(
            (i - 0.5) * (np.log((i - 0.5) / 2) - log_u)
            + (2 - i + 0.5) * (np.log((2 - i + 0.5) / 2) - log_v)
        ),
    }
    statistic = getattr(module, prefix + "InverseGammaGofStatistic")(
        InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 2})
    )
    assert statistic.execute_statistic(x) == pytest.approx(expected[prefix])
    with pytest.raises(TypeError, match="epsilon"):
        statistic.execute_statistic(x, epsilon=1e-10)


@pytest.mark.parametrize("cls", CLASSES)
def test_monte_carlo_reports_missing_generator(cls):
    with pytest.raises(ValueError, match="external calibration"):
        MonteCarloLimitDistributionResolver(2).resolve(cls(parameters_for(cls)), 5)


@pytest.mark.parametrize(
    "statistic",
    [
        module.LillieforsInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({})
        ),
        module.Chi2PearsonInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1})
        ),
        module.KolmogorovSmirnovInverseGammaGofStatistic(
            InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 1, "beta": 1}),
            alternative_type=AlternativeType.LEFT,
        ),
    ],
)
def test_unsafe_storage_calibration_is_blocked(statistic):
    store = Mock()
    with pytest.raises(ValueError):
        StorageLimitDistributionResolver(store).resolve(statistic, 10)
    store.get.assert_not_called()


def test_documented_examples():
    results = doctest.testmod(module)
    assert results.failed == 0
    assert results.attempted == 48
