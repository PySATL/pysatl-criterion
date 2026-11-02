"""Independent integrals, SciPy oracles and published reference representations.

References: docs/criteria/goodness-of-fit/exponentiality/mathematical-validation.md.
No oracle below calls a production kernel to generate its expected statistic.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import ExponentialDistributionDescriptor as E
from pysatl_criterion.hypothesis_testing.exponential_monte_carlo import monte_carlo_exponential
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import exponent as exp
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponent import (
    CoxOakesExponentialityGofStatistic,
    CramerVonMisesExponentialityGofStatistic,
    EppsPulleyExponentialityGofStatistic,
    KolmogorovSmirnovExponentialityGofStatistic,
    LillieforsExponentialityGofStatistic,
    MoranExponentialityGofStatistic,
)
from pysatl_criterion.utils.statistic import get_criterion_by_code


X = np.array([0.2, 0.5, 0.8, 1.4, 2.1])
NEW = [
    exp.AndersonDarlingExponentialityGofStatistic,
    exp.WatsonExponentialityGofStatistic,
    exp.KuiperExponentialityGofStatistic,
    exp.Chi2PearsonExponentialityGofStatistic,
    exp.GreenwoodExponentialityGofStatistic,
    exp.BaringhausHenzeELTExponentialityGofStatistic,
    exp.BaringhausHenzeBHCExponentialityGofStatistic,
]

EDF = [
    exp.AndersonDarlingExponentialityGofStatistic,
    exp.WatsonExponentialityGofStatistic,
    exp.KuiperExponentialityGofStatistic,
    exp.Chi2PearsonExponentialityGofStatistic,
    CramerVonMisesExponentialityGofStatistic,
]
TUNED = [
    exp.BaringhausHenzeELTExponentialityGofStatistic,
    exp.BaringhausHenzeBHCExponentialityGofStatistic,
]


@pytest.mark.parametrize("cls", NEW)
def test_contract_and_discovery(cls):
    obj = cls(E.DEFAULT.parse({}))
    assert obj.code() == f"{obj.short_code()}_EXPONENTIALITY_GOODNESS_OF_FIT"
    assert get_criterion_by_code(obj.code()) is cls
    assert obj.distribution() is E
    assert obj.hypothesis().parameters() == {}
    expected = (
        AlternativeType.TWO_TAILED
        if cls is exp.GreenwoodExponentialityGofStatistic
        else AlternativeType.RIGHT
    )
    assert obj.alternative().type() == expected
    before = X.copy()
    assert math.isfinite(obj.execute_statistic(X))
    np.testing.assert_array_equal(X, before)
    if cls not in EDF:
        with pytest.raises(TypeError):
            obj.execute_statistic(X, a=1)


@pytest.mark.parametrize("cls", NEW)
@pytest.mark.parametrize(
    "sample",
    [
        [],
        [1],
        [0, 0],
        [-1, 2],
        [[1, 2]],
        [1, np.nan],
        [1, np.inf],
        [1j, 2],
        np.ma.array([1, 2], mask=[1, 0]),
    ],
)
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls(E.DEFAULT.parse({})).execute_statistic(sample)


@pytest.mark.parametrize("cls", NEW)
@pytest.mark.parametrize("scale", [7.31, 1e-300, 1e300])
def test_scale_invariance_extremes_and_permutation(cls, scale):
    obj = cls(E.DEFAULT.parse({}))
    assert obj.execute_statistic(X[::-1] * scale) == pytest.approx(
        obj.execute_statistic(X), abs=2e-13
    )
    assert math.isfinite(obj.execute_statistic([1, 1]))


@pytest.mark.parametrize("cls", NEW)
def test_zero_boundary(cls):
    actual = cls(E.DEFAULT.parse({})).execute_statistic([0, 1, 2])
    if cls is exp.AndersonDarlingExponentialityGofStatistic:
        assert actual == math.inf
    else:
        assert math.isfinite(actual)


@pytest.mark.parametrize("cls", EDF)
def test_fixed_versus_estimated_and_rate_scale(cls):
    fixed = cls(E.DEFAULT.parse({"lam": 3}))
    fitted = cls(E.DEFAULT.parse({}))
    assert fixed.execute_statistic(X) != pytest.approx(fitted.execute_statistic(X))
    # Change units AND rate: a specified hypothesis should then be equivalent.
    assert cls(E.DEFAULT.parse({"lam": 3 / 7.31})).execute_statistic(X * 7.31) == pytest.approx(
        fixed.execute_statistic(X)
    )
    assert not math.isnan(fixed.execute_statistic([0]))
    for rate in (0, -1, np.inf, np.nan, True):
        with pytest.raises(ValueError):
            cls(E.DEFAULT.parse({"lam": rate}))


@pytest.mark.parametrize("rate", [None, 0.3, 3.0])
def test_edf_by_piecewise_integrals_and_scipy(rate):
    parameters = E.DEFAULT.parse({} if rate is None else {"lam": rate})
    scale = X.mean() if rate is None else 1 / rate
    u = stats.expon.cdf(X, scale=scale)
    cuts = np.r_[0, u, 1]
    mean_error = sum(
        integrate.quad(lambda t, k=k: k / len(X) - t, lo, hi)[0]
        for k, (lo, hi) in enumerate(pairwise(cuts))
    )
    ad = sum(
        integrate.quad(lambda t, k=k: (k / len(X) - t) ** 2 / (t * (1 - t)), lo, hi)[0]
        for k, (lo, hi) in enumerate(pairwise(cuts))
    ) * len(X)
    watson = sum(
        integrate.quad(lambda t, k=k: (k / len(X) - t - mean_error) ** 2, lo, hi)[0]
        for k, (lo, hi) in enumerate(pairwise(cuts))
    ) * len(X)
    cvm = stats.cramervonmises(X, stats.expon(scale=scale).cdf).statistic
    kuiper = sum(
        stats.kstest(X, stats.expon(scale=scale).cdf, alternative=tail).statistic
        for tail in ("less", "greater")
    )
    for cls, expected in [
        (exp.AndersonDarlingExponentialityGofStatistic, ad),
        (exp.WatsonExponentialityGofStatistic, watson),
        (exp.KuiperExponentialityGofStatistic, kuiper),
        (CramerVonMisesExponentialityGofStatistic, cvm),
    ]:
        assert cls(parameters).execute_statistic(X) == pytest.approx(expected, abs=2e-12)
    if rate is None:
        assert exp.AndersonDarlingExponentialityGofStatistic(parameters).execute_statistic(X) == (
            pytest.approx(stats.anderson(X, dist="expon").statistic)
        )


def test_pearson_scipy_histogram_oracle():
    rate = 3
    edges = stats.expon.ppf(np.linspace(0, 1, 6), scale=1 / rate)
    expected = stats.chisquare(np.histogram(X, bins=edges)[0]).statistic
    obj = exp.Chi2PearsonExponentialityGofStatistic(
        E.DEFAULT.parse({"lam": rate}), binning=EqualProbabilityBins(count=5)
    )
    assert obj.execute_statistic(X) == expected
    for bins in (True, 1, 2.5, np.nan):
        with pytest.raises((TypeError, ValueError)):
            EqualProbabilityBins(count=bins)
    obj = exp.Chi2PearsonExponentialityGofStatistic(
        E.DEFAULT.parse({}), binning=EqualProbabilityBins(count=7)
    )
    obj._validate_storage_calibration()
    assert (
        obj.configuration()["binning"]
        != exp.Chi2PearsonExponentialityGofStatistic(
            E.DEFAULT.parse({}), binning=EqualProbabilityBins(count=5)
        ).configuration()["binning"]
    )


@pytest.mark.parametrize("cls", TUNED)
@pytest.mark.parametrize("a", [0, -1, np.inf, np.nan, True, [], 101])
def test_invalid_tuning(cls, a):
    with pytest.raises(ValueError):
        cls(E.DEFAULT.parse({}), a=a)


@pytest.mark.parametrize("cls", TUNED)
def test_calibration_settings_not_lost(cls):
    with pytest.raises(ValueError, match="omits a"):
        cls(E.DEFAULT.parse({}), a=2)._validate_storage_calibration()
    cls(E.DEFAULT.parse({}))._validate_storage_calibration()


# Independent QUADPACK integrations of the defining integrals (not pair kernels).
# BH (1991), pp.552-553; Ebner (2023), equation (4). Values generated without PySATL.
@pytest.mark.parametrize(
    "a,elt,bhc",
    [
        (0.1, 0.20808196040736338, 0.18664761709683822),
        (1, 0.05095637603149553, 0.07645787651713727),
        (2, 0.022935774716549936, 0.03751769993746341),
        (10, 0.001239360334518456, 0.003765738333496624),
    ],
)
def test_transform_reference_values(a, elt, bhc):
    assert TUNED[0](E.DEFAULT.parse({}), a=a).execute_statistic(X) == pytest.approx(elt, abs=2e-13)
    assert TUNED[1](E.DEFAULT.parse({}), a=a).execute_statistic(X) == pytest.approx(bhc, abs=2e-13)


@pytest.mark.parametrize("a", [0.001, 0.1, 1, 10, 100])
def test_transforms_independent_integrals_and_author_representation(a):
    x = np.array([0, 0.1, 0.1, 0.2, 0.8, 3.1])
    y = x / x.mean()
    elt = (
        len(x)
        * integrate.quad(
            lambda t: np.mean((1 - (1 + t) * y) * np.exp(-t * y)) ** 2 * np.exp(-a * t),
            0,
            np.inf,
            epsabs=1e-11,
        )[0]
    )
    assert TUNED[0](E.DEFAULT.parse({}), a=a).execute_statistic(x) == pytest.approx(elt, abs=1e-10)
    # Alternative representation (5) / Appendix B R code of Ebner, not the
    # piecewise-polynomial production algorithm. Pair summation cancels at small a.
    if a >= 0.1:
        reference = 0.0
        for v in y:
            for w in y:
                m = min(v, w)
                reference += (
                    (abs(v - w) - 1) * math.exp(-a * m) / a
                    + (v + w - 2 * (m + 1)) * math.exp(-a * m) / a**2
                    - 2 * math.expm1(-a * m) / a**3
                )
        assert TUNED[1](E.DEFAULT.parse({}), a=a).execute_statistic(x) == pytest.approx(
            reference / len(x), abs=2e-11
        )


def test_greenwood_analytic_and_scipy_cv_identity():
    obj = exp.GreenwoodExponentialityGofStatistic(E.DEFAULT.parse({}))
    # Integer sample (2,5,8,14,21), sum=50, sum of squares=730.
    assert obj.execute_statistic(X) == pytest.approx(730 / 2500)
    assert obj.execute_statistic(X) == pytest.approx((1 + stats.variation(X) ** 2) / len(X))


def test_signed_contrasts_reference_values_and_both_tails():
    # Product y_i=0.2352; Moran's statistic is gamma+log(product)/5.
    # Cox-Oakes = 5 + log(.2**.8*.5**.5*.8**.2/(1.4**.4*2.1**1.1)).
    references = [
        (MoranExponentialityGofStatistic, np.euler_gamma + math.log(0.2352) / 5),
        (
            CoxOakesExponentialityGofStatistic,
            5 + math.log(0.2**0.8 * 0.5**0.5 * 0.8**0.2 / (1.4**0.4 * 2.1**1.1)),
        ),
        (
            EppsPulleyExponentialityGofStatistic,
            math.sqrt(240) * (math.fsum(map(math.exp, -X)) / 5 - 0.5),
        ),
    ]
    for cls, value in references:
        obj = cls(E.DEFAULT.parse({}))
        assert obj.execute_statistic(X) == pytest.approx(value)
        assert obj.alternative().type() == AlternativeType.TWO_TAILED


@pytest.mark.parametrize("cls", NEW + [MoranExponentialityGofStatistic])
def test_reproducible_monte_carlo(cls):
    obj = cls(E.DEFAULT.parse({}))
    first = monte_carlo_exponential(obj, X, n_resamples=29, random_state=173)
    assert first == monte_carlo_exponential(obj, X, n_resamples=29, random_state=173)
    assert 0 < first.p_value <= 1
    assert first.rejected == (first.p_value <= 0.05)


def test_monte_carlo_rank_oracle_and_known_rate():
    obj = KolmogorovSmirnovExponentialityGofStatistic(E.DEFAULT.parse({"lam": 3}))
    rng = np.random.default_rng(932)
    null = [
        stats.kstest(rng.exponential(1 / 3, 5), stats.expon(scale=1 / 3).cdf).statistic
        for _ in range(99)
    ]
    observed = stats.kstest(X, stats.expon(scale=1 / 3).cdf).statistic
    result = monte_carlo_exponential(obj, X, n_resamples=99, random_state=932)
    assert result.p_value == (1 + sum(t >= observed for t in null)) / 100
    fitted = LillieforsExponentialityGofStatistic(E.DEFAULT.parse({}))
    assert (
        monte_carlo_exponential(fitted, X, n_resamples=99, random_state=932).p_value
        != result.p_value
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"n_resamples": 0},
        {"n_resamples": True},
        {"n_resamples": 2.5},
        {"significance_level": np.nan},
        {"significance_level": 1},
        {"significance_level": True},
    ],
)
def test_invalid_monte_carlo_settings(settings):
    with pytest.raises(ValueError):
        monte_carlo_exponential(TUNED[0](E.DEFAULT.parse({})), X, **settings)


@pytest.mark.parametrize("cls", NEW + [CramerVonMisesExponentialityGofStatistic])
def test_composite_scale_roundtrip_and_minimum(cls):
    obj = cls(E.DEFAULT.parse({}))
    assert math.isfinite(obj.execute_statistic([0.3, 0.7]))
    assert obj.execute_statistic(X * 7.31) == pytest.approx(obj.execute_statistic(X))


def test_greenwood_and_we_are_affine_equivalent():
    from pysatl_criterion.statistics.goodness_of_fit.continuous.exponent import (
        WeExponentialityGofStatistic,
    )

    g = exp.GreenwoodExponentialityGofStatistic(E.DEFAULT.parse({})).execute_statistic(X)
    we_value = WeExponentialityGofStatistic(E.DEFAULT.parse({})).execute_statistic(X)
    n = len(X)
    assert we_value == pytest.approx(g - 1 / n)


def test_lilliefors_independent_fitted_scipy_statistic():
    obj = LillieforsExponentialityGofStatistic(E.DEFAULT.parse({}))
    expected = stats.kstest(X, stats.expon(scale=X.mean()).cdf).statistic
    # Compare only D: SciPy's simple-null p-value here would be invalid.
    assert obj.execute_statistic(X) == pytest.approx(expected)
    assert obj.execute_statistic(7.31 * X) == pytest.approx(expected)


@pytest.mark.parametrize(
    "name,settings,expected",
    [
        ("GiniExponentialityGofStatistic", {}, 47 / 100),
        ("GnedenkoExponentialityGofStatistic", {"r": 2}, 33 / 28),
        ("HarrisExponentialityGofStatistic", {"r": 1}, 17 / 22),
    ],
)
def test_legacy_analytic_references(name, settings, expected):
    from pysatl_criterion.statistics.goodness_of_fit.continuous import exponent

    # Gini: weighted order sum =94/200. Normalized order spacings are
    # (1, 1.2, .9, 1.2, .7): first two / last three means =33/28;
    # outer two / inner three means =17/22. Independent rational arithmetic.
    obj = getattr(exponent, name)(E.DEFAULT.parse({}), **settings)
    assert obj.execute_statistic(X) == pytest.approx(expected)
