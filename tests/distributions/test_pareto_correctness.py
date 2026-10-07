"""Independent formulas and contract regressions for all Pareto statistics."""

import doctest
import inspect
from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import pareto


CLASSES = [
    cls
    for name, cls in vars(pareto).items()
    if name.endswith("ParetoGofStatistic") and not name.startswith("Abstract")
]
FIXED = [
    pareto.KolmogorovSmirnovParetoGofStatistic,
    pareto.AndersonDarlingParetoGofStatistic,
    pareto.CramerVonMisesParetoGofStatistic,
    pareto.MinToshiyukiParetoGofStatistic,
]
FITTED = [
    pareto.LillieforsParetoGofStatistic,
    pareto.GreenwoodParetoGofStatistic,
    pareto.LequesneKlParetoGofStatistic,
]


@pytest.mark.parametrize("cls", CLASSES)
def test_interface_and_repeated_calls(cls):
    assert not inspect.isabstract(cls)
    stat: AbstractGoodnessOfFitStatistic = cls()
    x = np.array([3.1, 1.2, 8.3, 2.7, 1.9, 4.6])
    original = x.copy()
    state = vars(stat).copy()
    null = stat.hypothesis().parameters()
    first = stat.execute_statistic(x, unused=True)
    assert isinstance(first, (float, np.float64))
    stat.execute_statistic([2, 3.1, 4.3, 8.1, 10.7])
    assert stat.execute_statistic(x) == first
    np.testing.assert_array_equal(x, original)
    assert vars(stat) == state
    assert stat.hypothesis().parameters() == null
    expected_tail = (
        AlternativeType.TWO_TAILED
        if cls == pareto.GreenwoodParetoGofStatistic
        else AlternativeType.RIGHT
    )
    assert stat.alternative().type() == expected_tail


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "x",
    [
        [],
        [np.nan, 2, 3, 4],
        [1, 2, np.inf, 4],
        [[1, 2], [3, 4]],
        [1 + 0j, 2, 3, 4],
        [0, 2, 3, 4],
        [-1, 2, 3, 4],
    ],
)
def test_invalid_samples(cls, x):
    with pytest.raises(ValueError):
        cls().execute_statistic(x)


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize("param", ["shape", "scale"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, [1], 1j, None])
def test_invalid_fixed_parameters(cls, param, value):
    with pytest.raises(ValueError):
        cls(**{param: value})


@pytest.mark.parametrize("cls", FIXED)
def test_fixed_support_and_constant(cls):
    with pytest.raises(ValueError, match="at least scale"):
        cls(scale=2).execute_statistic([1.9, 2, 3, 4])
    assert np.isfinite(cls().execute_statistic([2, 2, 2, 2]))


@pytest.mark.parametrize(
    "direction,scipy_direction",
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_scipy_and_tail(direction, scipy_direction):
    x = [2.1, 2.3, 3.7, 4.9, 7.2]
    stat = pareto.KolmogorovSmirnovParetoGofStatistic(direction, shape=1.7, scale=2)
    expected = stats.kstest(
        x, stats.pareto(1.7, scale=2).cdf, alternative=scipy_direction
    ).statistic
    assert stat.execute_statistic(x) == pytest.approx(expected)
    assert stat.alternative().type() == AlternativeType.RIGHT


def test_ad_cvm_mt_independent_formulas():
    x = np.array([2.1, 2.8, 4.7, 6.2, 9.1])
    u = stats.pareto(2.3, scale=2).cdf(x)
    n = len(x)
    i = np.arange(1, n + 1)
    ad = -n - np.dot(2 * i - 1, np.log(u) + np.log1p(-u[::-1])) / n
    cvm = stats.cramervonmises(x, stats.pareto(2.3, scale=2).cdf).statistic
    mt = sum(
        max(j / n - v, v - (j - 1) / n) / np.sqrt(v * (1 - v)) for j, v in enumerate(u, 1)
    ) / np.sqrt(n)
    for cls, expected in zip(FIXED[1:], [ad, cvm, mt], strict=True):
        assert cls(shape=2.3, scale=2).execute_statistic(x) == pytest.approx(expected)


@pytest.mark.parametrize(
    "cls", [pareto.AndersonDarlingParetoGofStatistic, pareto.MinToshiyukiParetoGofStatistic]
)
def test_true_boundary_infinity_and_large_tail(cls):
    assert cls().execute_statistic([1, 2]) == np.inf
    # F(1e20) rounds to one, but its survival and this statistic are finite.
    assert np.isfinite(cls().execute_statistic([2, 1e20]))
    # A ratio exceeding float64 range still has a representable logarithm.
    assert np.isfinite(cls(shape=0.01, scale=1e-300).execute_statistic([1e-200, 1e300]))


@pytest.mark.parametrize("cls", FITTED)
def test_fits_and_scale_invariance(cls):
    stat = cls()
    x = np.array([1.1, 1.7, 2.3, 5.1, 8.7, 19.2])
    expected = stat.execute_statistic(x)
    for scale in [1e-250, 1e250]:
        assert stat.execute_statistic(x * scale) == pytest.approx(expected, abs=1e-12)
    assert stat.hypothesis().parameters() == {}
    with pytest.raises(ValueError, match="nonconstant"):
        stat.execute_statistic([2] * 6)


@pytest.mark.parametrize("cls", FITTED[:2])
def test_power_invariance(cls):
    x = np.array([1.1, 1.7, 2.3, 5.1, 8.7, 19.2])
    assert cls().execute_statistic(x**3) == pytest.approx(cls().execute_statistic(x))


def test_greenwood_formula():
    # Logarithmic residuals are proportional to (0, 1, 2, 3).
    assert pareto.GreenwoodParetoGofStatistic().execute_statistic([1, 2, 4, 8]) == pytest.approx(
        14 / 36
    )
    with pytest.raises(ValueError):
        pareto.GreenwoodParetoGofStatistic().execute_statistic([1, 2])


@pytest.mark.parametrize(
    "x", [[1, 2, 2, 4], [1, 5, 25, 125], [1.2, 1.9, 3.1, 4.7], [2] * 4, [1] * 4]
)
@pytest.mark.parametrize("scale", [1, 3])
def test_obradovic_inclusive_indicators(x, scale):
    z = np.array(x, dtype=float)
    ratios = [max(a / b, b / a) for a, b in combinations(z, 2)]
    expected = np.mean([np.mean(np.array(ratios) <= t) - np.mean(z <= t) for t in z])
    stat = pareto.ObradovicParetoGofStatistic(scale=scale)
    assert stat.execute_statistic(z * scale) == pytest.approx(expected)
    assert stat.hypothesis().parameters() == {"scale": scale}


@pytest.mark.parametrize("m", [1, 2, 3])
def test_kl_independent_density_entropy(m):
    x = np.array([1.1, 1.3, 1.8, 2.4, 4, 8, 17])
    n = len(x)
    shape = 1 / np.mean(np.log(x / x[0]))
    entropy = (
        sum(np.log(n / (2 * m) * (x[min(i + m, n - 1)] - x[max(i - m, 0)])) for i in range(n)) / n
    )
    expected = -entropy - np.mean(stats.pareto.logpdf(x, shape, scale=x[0]))
    assert pareto.LequesneKlParetoGofStatistic().execute_statistic(x, m=m) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("m", [0, -1, 3, 5, 1.2, True, np.nan, [1]])
def test_invalid_window(m):
    with pytest.raises(ValueError, match="m must"):
        pareto.LequesneKlParetoGofStatistic().execute_statistic([1, 2, 3, 4, 5, 6], m=m)


def test_zero_spacing_and_signed_kl():
    stat = pareto.LequesneKlParetoGofStatistic()
    assert stat.execute_statistic([1, 1, 1, 2, 3, 4], m=1) == np.inf
    # Widely separated observations can give a negative fixed-window estimate.
    x = np.exp(np.arange(10, dtype=float))
    expected = -stats.differential_entropy(x, window_length=3, method="vasicek")
    shape = 1 / np.mean(np.log(x))
    expected -= np.mean(stats.pareto.logpdf(x, shape))
    assert expected < 0
    assert stat.execute_statistic(x, m=3) == pytest.approx(expected)
    assert stat.execute_statistic(x**2, m=3) != pytest.approx(expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_missing_simulator_is_explicit(cls):
    with pytest.raises(ValueError, match="Pareto requires external calibration"):
        MonteCarloLimitDistributionResolver(2).resolve(cls(), 10)


def test_storage_guards(mocker):
    store = mocker.Mock()
    resolver = StorageLimitDistributionResolver(store)
    with pytest.raises(ValueError, match="shape-specific"):
        resolver.resolve(pareto.LequesneKlParetoGofStatistic(), 10)
    with pytest.raises(ValueError, match="direction"):
        resolver.resolve(pareto.KolmogorovSmirnovParetoGofStatistic(AlternativeType.LEFT), 10)
    store.get.assert_not_called()


def test_documented_examples():
    assert doctest.testmod(pareto).failed == 0


def test_obradovic_overflowing_ratios():
    # Independently count comparisons with exact rational arithmetic.
    from fractions import Fraction

    x = [1e-300, 1e-100, 1e100, 1e300]
    scale = 1e-300
    z = [Fraction(t) / Fraction(scale) for t in x]
    ratios = [max(a / b, b / a) for a, b in combinations(z, 2)]
    expected = sum(
        sum(r <= t for r in ratios) / len(ratios) - sum(v <= t for v in z) / len(z) for t in z
    ) / len(z)
    assert pareto.ObradovicParetoGofStatistic(scale).execute_statistic(x) == pytest.approx(expected)


@pytest.mark.parametrize("cls", FITTED)
def test_adjacent_large_observations(cls):
    x = 1e300 + np.arange(8) * np.spacing(1e300)
    assert np.isfinite(cls().execute_statistic(x))


def test_mt_underflowing_probability_has_finite_penalty():
    shape = np.nextafter(0.0, 1.0)
    x = np.nextafter(1.0, 2.0)
    # n=1: d=1-u, u approximately shape*log(x); compute in log space.
    expected = np.exp(-0.5 * (np.log(shape) + np.log(np.log1p(x - 1))))
    value = pareto.MinToshiyukiParetoGofStatistic(shape=shape).execute_statistic([x])
    assert np.isfinite(value)
    assert value == pytest.approx(expected)
