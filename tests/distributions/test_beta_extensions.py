"""Independent polynomial/integral references and fitted-Beta calibration checks."""

from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.hypothesis_testing.beta_bootstrap import parametric_bootstrap_beta
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit import (
    EbnerLiebenbergBetaGofStatistic,
    NeymanSmoothBetaGofStatistic,
    RaschkeBetaGofStatistic,
)
from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code


SAMPLE = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
FITTED = [EbnerLiebenbergBetaGofStatistic, RaschkeBetaGofStatistic]


@pytest.mark.parametrize(
    "cls,code,fixed",
    [
        (NeymanSmoothBetaGofStatistic, "NEYMAN", {"a": 2, "b": 5}),
        (RaschkeBetaGofStatistic, "RASCHKE", {}),
        (EbnerLiebenbergBetaGofStatistic, "EL", {}),
    ],
)
def test_metadata_and_discovery(cls, code, fixed):
    statistic = cls(Beta.DEFAULT.parse(fixed))
    assert cls.short_code() == code
    assert cls.code() == f"{code}_BETA_GOODNESS_OF_FIT"
    assert cls.distribution() is Beta
    assert isinstance(statistic.alternative(), RightAlternative)
    assert statistic.hypothesis().parameters() == fixed
    assert cls in get_available_criteria(DistributionType.BETA)
    assert get_criterion_by_code(cls.code()) is cls


@pytest.mark.parametrize("a,b", [(0.1, 0.2), (1, 1), (2, 5), (80, 120)])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_neyman_explicit_polynomial_reference(a, b, k):
    x = stats.beta.ppf(np.linspace(0.05, 0.95, 15), a, b)
    # Independently integrate the density instead of using the production CDF.
    u = np.array([integrate.quad(lambda t: stats.beta.pdf(t, a, b), 0, v)[0] for v in x])
    polynomials = [
        np.sqrt(3) * (2 * u - 1),
        np.sqrt(5) * (6 * u**2 - 6 * u + 1),
        np.sqrt(7) * (20 * u**3 - 30 * u**2 + 12 * u - 1),
        3 * (70 * u**4 - 140 * u**3 + 90 * u**2 - 20 * u + 1),
    ]
    expected = sum(np.mean(p) ** 2 for p in polynomials[:k]) * len(x)
    statistic = NeymanSmoothBetaGofStatistic(Beta.DEFAULT.parse({"a": a, "b": b}), k=k)
    assert statistic.execute_statistic(x) == pytest.approx(expected, abs=1e-8)


def test_neyman_higher_order_exact_endpoint_reference():
    # P_j(1)=1: for a single endpoint the sum is k*(k+2), all orders.
    statistic = NeymanSmoothBetaGofStatistic(Beta.DEFAULT.parse({"a": 0.01, "b": 100}), k=9)
    assert statistic.execute_statistic([1]) == pytest.approx(99)
    assert statistic.execute_statistic([0]) == pytest.approx(99)
    assert np.isfinite(
        statistic.execute_statistic([np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)])
    )


@pytest.mark.parametrize("k", [0, -1, 1.5, True, np.bool_(True), np.nan, "4"])
def test_neyman_invalid_order(k):
    with pytest.raises(ValueError):
        NeymanSmoothBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}), k=k)


@pytest.mark.parametrize("name", ["a", "b"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, 1j])
def test_invalid_shapes(name, value):
    fixed = {"a": 2, "b": 5, name: value}
    with pytest.raises((ValueError, TypeError)):
        NeymanSmoothBetaGofStatistic(Beta.DEFAULT.parse(fixed))


@pytest.mark.parametrize("cls", FITTED)
@pytest.mark.parametrize("fixed", [{"a": 2}, {"b": 3}, {"a": 2, "b": 3}])
def test_fitted_classes_reject_fixed_shapes(cls, fixed):
    with pytest.raises(ValueError, match="Unsupported"):
        cls(Beta.DEFAULT.parse(fixed))


@pytest.mark.parametrize("cls", [NeymanSmoothBetaGofStatistic, RaschkeBetaGofStatistic])
@pytest.mark.parametrize(
    "x",
    [
        [],
        [-0.1, 0.5],
        [0.5, 1.1],
        [np.nan],
        [np.inf],
        [[0.1, 0.2]],
        [0.1 + 1j],
        np.ma.array([0.1, 0.2], mask=[True, False]),
    ],
)
def test_invalid_samples(cls, x):
    fixed = {"a": 2, "b": 5} if cls is NeymanSmoothBetaGofStatistic else {}
    with pytest.raises(ValueError):
        cls(Beta.DEFAULT.parse(fixed)).execute_statistic(x)


@pytest.mark.parametrize("x", [[0.2], [0.1, 0.4], [0.2] * 4, [0, 0.2, 0.4], [0.2, 0.5, 1]])
def test_raschke_fit_constraints(x):
    with pytest.raises(ValueError):
        RaschkeBetaGofStatistic(Beta.DEFAULT.parse({})).execute_statistic(x)


def test_raschke_independent_ad_integral():
    x = np.array(SAMPLE)
    a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    y = stats.norm.ppf(stats.beta.cdf(x, a, b))
    mean, scale = stats.norm.fit(y)
    u = np.sort(stats.norm.cdf(y, loc=mean, scale=scale))
    edges = np.r_[0, u, 1]
    # Definition of weighted EDF distance; no AD summation kernel.
    value = 0.0
    n = len(x)
    for i, (lo, hi) in enumerate(pairwise(edges)):
        value += integrate.quad(lambda t, i=i: n * (i / n - t) ** 2 / (t * (1 - t)), lo, hi)[0]
    expected = value * (1 + 0.75 / n + 2.25 / n**2)
    statistic = RaschkeBetaGofStatistic(Beta.DEFAULT.parse({}))
    assert statistic.execute_statistic(x) == pytest.approx(expected, rel=1e-10)
    assert statistic.execute_statistic(x[::-1]) == pytest.approx(expected, rel=1e-10)
    np.testing.assert_array_equal(x, SAMPLE)


@pytest.mark.parametrize("cls", FITTED)
@pytest.mark.parametrize("x", [[1e-14, 0.02, 0.2, 0.7, 1 - 1e-14], [0.1, 0.4, 0.8]])
def test_fitted_near_boundaries(cls, x):
    value = cls(Beta.DEFAULT.parse({})).execute_statistic(x)
    assert np.isfinite(value) and value >= 0


@pytest.mark.parametrize("cls", FITTED)
def test_bootstrap_reproducibility_and_refitting(cls, mocker):
    statistic = cls(Beta.DEFAULT.parse({}))
    spy = mocker.spy(statistic, "execute_statistic")
    fit_spy = mocker.spy(statistic, "_fit")
    result = parametric_bootstrap_beta(statistic, SAMPLE, n_resamples=19, random_state=81)
    assert spy.call_count == 20
    assert fit_spy.call_count == 21  # observed, generating fit, and all 19 replicates

    # SciPy supplies an independent simulation and refitting driver.
    def evaluate(fitted_distribution, data, axis):
        return np.apply_along_axis(statistic.execute_statistic, axis, data)

    reference = stats.goodness_of_fit(
        stats.beta,
        SAMPLE,
        known_params={"loc": 0, "scale": 1},
        statistic=evaluate,
        n_mc_samples=19,
        random_state=np.random.default_rng(81),
    )
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.critical_value == pytest.approx(
        np.quantile(reference.null_distribution, 0.95, method="inverted_cdf")
    )
    assert result.rejected == (result.statistic > result.critical_value)
    assert result == parametric_bootstrap_beta(statistic, SAMPLE, n_resamples=19, random_state=81)
    assert statistic.hypothesis().parameters() == {}


@pytest.mark.parametrize(
    "options",
    [
        {"n_resamples": 0},
        {"n_resamples": True},
        {"n_resamples": 2.5},
        {"significance_level": 0},
        {"significance_level": 1},
        {"significance_level": np.nan},
        {"significance_level": True},
    ],
)
def test_bootstrap_invalid_settings(options):
    statistic = EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))
    with pytest.raises(ValueError):
        parametric_bootstrap_beta(statistic, SAMPLE, **options)


def test_bootstrap_does_not_silently_drop_failures(mocker):
    statistic = EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))
    mocker.patch.object(statistic, "execute_statistic", side_effect=[1.0, ValueError("fit failed")])
    with pytest.raises(ValueError, match="fit failed"):
        parametric_bootstrap_beta(statistic, SAMPLE, n_resamples=3, random_state=42)
