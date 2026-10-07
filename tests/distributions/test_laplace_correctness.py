"""Independent mathematical and public-contract regressions for fixed Laplace nulls."""

import doctest
from itertools import pairwise
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import laplace

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import laplace as module


CLASSES = [
    module.KolmogorovSmirnovLaplaceGofStatistic,
    module.CramerVonMisesLaplaceGofStatistic,
    module.AndersonDarlingLaplaceGofStatistic,
    module.KuiperLaplaceGofStatistic,
    module.WatsonLaplaceGofStatistic,
    module.GreenwoodLaplaceGofStatistic,
]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "sample", [[], 1.0, [[1, 2]], [np.nan], [np.inf], [-np.inf], [1j], ["1"], [None], [[1], [2, 3]]]
)
def test_invalid_sample(cls, sample):
    with pytest.raises(ValueError):
        cls().execute_statistic(sample)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("name", ["t", "s"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [1], 1j, "1", None])
def test_invalid_parameters(cls, name, value):
    with pytest.raises(ValueError):
        cls(**{name: value})


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("sample", [[0.0], [0.0, 0.0], [-2.0, 0.5, 0.5, 3.0]])
def test_contract_invariance_and_call_independence(cls, sample):
    x = np.asarray(sample)
    original = x.copy()
    statistic = cls(t=3, s=2)
    expected = cls().execute_statistic(x)
    y = 3 + 2 * x
    result = statistic.execute_statistic(y, compatibility_keyword=True)
    assert isinstance(result, (float, np.float64))
    assert result == pytest.approx(expected)
    statistic.execute_statistic([-30.0, 50.0])
    assert statistic.execute_statistic(y) == result
    assert statistic.hypothesis().parameters() == {"t": 3.0, "s": 2.0}
    assert statistic.alternative().type() == AlternativeType.RIGHT
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(y, 3 + 2 * x)


@pytest.mark.parametrize("cls", CLASSES)
def test_standardization_avoids_intermediate_overflow(cls):
    assert cls(t=-1e308, s=1e308).execute_statistic([1e308]) == pytest.approx(
        cls().execute_statistic([2.0])
    )


@pytest.mark.parametrize("direction", list(AlternativeType))
def test_ks_directions(direction):
    sample = [-2, -1, 0.5, 0.5]
    u = laplace.cdf(sample)
    plus = max((i + 1) / len(u) - value for i, value in enumerate(u))
    minus = max(value - i / len(u) for i, value in enumerate(u))
    expected = {
        AlternativeType.RIGHT: plus,
        AlternativeType.LEFT: minus,
        AlternativeType.TWO_TAILED: max(plus, minus),
    }[direction]
    statistic = module.KolmogorovSmirnovLaplaceGofStatistic(alternative_type=direction)
    assert statistic.execute_statistic(sample) == pytest.approx(expected)
    assert statistic.alternative().type() == AlternativeType.RIGHT


@pytest.mark.parametrize("setting", [{"alternative_type": "right"}, {"mode": "bad"}])
def test_invalid_ks_settings(setting):
    with pytest.raises(ValueError):
        module.KolmogorovSmirnovLaplaceGofStatistic(**setting)


@pytest.mark.parametrize("mode", ["auto", "exact", "approx", "asymp"])
def test_ks_mode_is_compatibility_only(mode):
    statistic = module.KolmogorovSmirnovLaplaceGofStatistic(mode=mode)
    assert statistic.execute_statistic([0.0]) == 0.5


@pytest.mark.parametrize("direction", [AlternativeType.LEFT, AlternativeType.RIGHT])
def test_one_sided_ks_rejects_ambiguous_storage_but_allows_monte_carlo(direction):
    statistic = module.KolmogorovSmirnovLaplaceGofStatistic(direction, t=3, s=2)
    store = Mock()
    with pytest.raises(ValueError, match="CDF direction"):
        StorageLimitDistributionResolver(store).resolve(statistic, 5)
    store.get.assert_not_called()
    results = MonteCarloLimitDistributionResolver(3).resolve(statistic, 5)
    assert len(results) == 3
    assert np.all(np.isfinite(results))


def test_two_sided_ks_storage_retains_fixed_parameters():
    store = Mock()
    store.get.return_value = None
    statistic = module.KolmogorovSmirnovLaplaceGofStatistic(t=3, s=2)
    assert StorageLimitDistributionResolver(store).resolve(statistic, 5) is None
    store.get.assert_called_once()


def test_ad_far_tails_analytic_reference():
    # At (-1000, 0, 1000), tail logs have limits -1000-log(2), -log(2), 0.
    expected = -3 + 2 * (1000 + np.log(2)) / 3 + 2 * np.log(2)
    result = module.AndersonDarlingLaplaceGofStatistic().execute_statistic([-1000, 0, 1000])
    assert np.isfinite(result)
    assert result == pytest.approx(expected, rel=1e-14)


def test_ad_does_not_overflow_log_pair_before_weighting():
    result = module.AndersonDarlingLaplaceGofStatistic().execute_statistic([-1e308, 0, 1e308])
    assert np.isfinite(result)
    assert result == pytest.approx((1e308 / 3) * 2)


def test_ad_float64_range_limit_is_not_clipped():
    assert np.isposinf(
        module.AndersonDarlingLaplaceGofStatistic(s=1e-308).execute_statistic([1e308])
    )


@pytest.mark.parametrize(
    "cls,weight,center",
    [
        (module.CramerVonMisesLaplaceGofStatistic, False, False),
        (module.AndersonDarlingLaplaceGofStatistic, True, False),
        (module.WatsonLaplaceGofStatistic, False, True),
    ],
)
def test_quadratic_edf_statistics_against_integral(cls, weight, center):
    # Integrate defining EDF discrepancy over probability space, interval by interval.
    x = [-3.0, -0.2, 0.7, 2.0]
    u = laplace.cdf(x)
    edges = np.r_[0.0, u, 1.0]
    n = len(x)
    mean = sum(
        quad(lambda v, i=i: i / n - v, left, right)[0]
        for i, (left, right) in enumerate(pairwise(edges))
    )
    expected = 0.0
    for i, (left, right) in enumerate(pairwise(edges)):

        def integrand(v, i=i):
            difference = i / n - v - (mean if center else 0.0)
            return difference**2 / (v * (1 - v)) if weight else difference**2

        expected += n * quad(integrand, left, right)[0]
    assert cls().execute_statistic(x) == pytest.approx(expected, abs=1e-12)


def test_greenwood_endpoint_spacings_and_ties():
    # u = [1/4, 1/2, 1/2, 3/4], spacings = [1/4, 1/4, 0, 1/4, 1/4].
    x = [-np.log(2), 0, 0, np.log(2)]
    assert module.GreenwoodLaplaceGofStatistic().execute_statistic(x) == pytest.approx(0.25)


def test_watson_centering_and_singleton_degeneracy():
    statistic = module.WatsonLaplaceGofStatistic()
    for x in [-1000, 0, 1000]:
        assert statistic.execute_statistic([x]) == pytest.approx(1 / 12)
    # Equally spaced midpoint PIT values give the exact lower bound 1/(12n).
    n = 1000
    x = laplace.ppf((np.arange(n) + 0.5) / n)
    assert statistic.execute_statistic(x) == pytest.approx(1 / (12 * n), abs=1e-15)


def test_documented_examples():
    assert doctest.testmod(module).failed == 0
