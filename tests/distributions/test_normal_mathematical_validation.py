"""Independent numerical definitions for the normality audit.

These references do not call production EDF/frequency kernels. Published-source
qualifications are recorded in normality/mathematical-validation.md.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, special, stats

from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import normal_statistics
from tests.parameter_cases import parameters_for


SAMPLES = [
    np.random.default_rng(41).normal(size=24),
    np.random.default_rng(42).exponential(size=31),
    np.array([-3, -2, -1, -1, 0, 0, 0, 1, 1, 2, 3, 4.0]),
]


def value(name, x, **kwargs):
    cls = getattr(normal, name + "NormalityGofStatistic")
    return cls(parameters_for(cls), **kwargs).execute_statistic(x)


@pytest.mark.parametrize("x", SAMPLES)
@pytest.mark.parametrize("name", normal_statistics._EDF_STATISTICS)
def test_edf_against_independent_definitions(name, x):  # noqa: C901 - explicit reference catalog
    cls = getattr(normal, name)
    criterion = cls(parameters_for(cls))
    z = np.sort(x)
    if not criterion.hypothesis().parameters():
        z = (z - z.mean()) / z.std(ddof=1)
    n = len(z)
    # Integrate the density independently of the production normal CDF.
    u = np.array(
        [
            integrate.quad(
                lambda t: np.exp(-t * t / 2) / np.sqrt(2 * np.pi), -np.inf, t, epsabs=1e-13
            )[0]
            for t in z
        ]
    )
    ranks = np.arange(1, n + 1)
    dplus = max(ranks / n - u)
    dminus = max(u - (ranks - 1) / n)
    short = cls.short_code()
    if short in ("KS", "LILLIE"):
        expected = max(dplus, dminus)
    elif short == "KS_PLUS":
        expected = dplus
    elif short == "KS_MINUS":
        expected = dminus
    elif short in ("CVM", "AD", "WATSON"):
        edges = np.r_[0, u, 1]
        expected = 0.0
        center = 0.5 - u.mean() if short == "WATSON" else 0
        for j, (low, high) in enumerate(pairwise(edges)):

            def integrand(t, j=j):
                residual = j / n - t - center
                return residual**2 / (t * (1 - t)) if short == "AD" else residual**2

            expected += n * integrate.quad(integrand, low, high, epsabs=1e-12)[0]
    elif "Kuiper" in name:
        factor = 1
        if "Scaled" in name:
            factor = np.sqrt(n)
        elif "Corrected" in name:
            factor = np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)
        expected = factor * (dplus + dminus)
    elif "Liao" in name:
        expected = sum(
            max(i / n - p, p - (i - 1) / n) / np.sqrt(p * (1 - p)) for i, p in enumerate(u, 1)
        ) / np.sqrt(n)
    elif short in ("ZHANG_ZA", "ZWA"):
        expected = -sum(
            np.log(p) / (n - i + 0.5) + np.log1p(-p) / (i - 0.5) for i, p in enumerate(u, 1)
        )
        if short == "ZWA":
            expected = 10 * expected - 32
    elif short in ("ZHANG_ZC", "ZWC"):
        expected = sum(
            np.log((1 - p) / p * (i - 0.75) / (n - i + 0.25)) ** 2 for i, p in enumerate(u, 1)
        )
    elif short == "ZHANG_ZK":
        expected = max(
            (i - 0.5) * np.log((i - 0.5) / (n * p))
            + (n - i + 0.5) * np.log((n - i + 0.5) / (n * (1 - p)))
            for i, p in enumerate(u, 1)
        )
    else:
        pytest.fail(f"Missing independent reference for {name}")
    assert criterion.execute_statistic(x) == pytest.approx(expected, rel=2e-9, abs=2e-10)


@pytest.mark.parametrize("x", SAMPLES)
@pytest.mark.parametrize(
    "name,lambda_",
    [
        ("Chi2Pearson", 1),
        ("FreemanTukey", -0.5),
        ("ModifiedLogLikelihood", -1),
        ("NeymanChi2", -2),
        ("CressieRead", 2 / 3),
    ],
)
def test_frequency_against_scipy_power_divergence(x, name, lambda_):
    edges = np.r_[-np.inf, np.quantile(x, [0.25, 0.5, 0.75]), np.inf]
    observed = np.histogram(x, edges)[0]
    expected = len(x) * np.diff(stats.norm.cdf(edges, loc=x.mean(), scale=x.std(ddof=1)))
    reference = stats.power_divergence(observed, expected, lambda_=lambda_).statistic
    assert value(name, x, binning=ExplicitBins(edges)) == pytest.approx(reference, rel=1e-11)


@pytest.mark.parametrize("name", ["ModifiedLogLikelihood", "NeymanChi2"])
def test_empty_frequency_cell_has_infinite_discrepancy(name):
    assert np.isposinf(value(name, SAMPLES[0], binning=ExplicitBins([-np.inf, 5, np.inf])))


@pytest.mark.parametrize("x", SAMPLES)
def test_probability_plot_statistics(x):
    n = len(x)
    ordered = np.sort(x)
    scores = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    corr = np.corrcoef(ordered, scores)[0, 1]
    assert value("LooneyGulledge", x) == pytest.approx(corr)
    assert value("RyanJoiner", x) == pytest.approx(corr)
    assert value("SF", x) == pytest.approx(corr**2)
    assert value("Filli", x) == pytest.approx(stats.probplot(x)[1][2])
    for alpha in ("0", "1/2", "3/8"):
        a = {"0": 0, "1/2": 0.5, "3/8": 0.375}[alpha]
        positions = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
        for observation in np.unique(ordered):
            tied = ordered == observation
            positions[tied] = positions[tied].mean()
        expected = np.corrcoef(ordered, stats.norm.ppf(positions))[0, 1]
        assert value("RyanJoiner", x, weighted=True, cte_alpha=alpha) == pytest.approx(expected)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 20, 100, 1000])
def test_shapiro_wilk_royston_coefficients(n):
    x = np.random.default_rng(n).exponential(size=n)
    assert value("ShapiroWilk", x) == pytest.approx(stats.shapiro(x).statistic, abs=2e-8)


@pytest.mark.parametrize("x", SAMPLES)
def test_hermite_statistics_against_special_function(x):
    z = (x - x.mean()) / x.std(ddof=1)
    for name, degree in (("BontempsMeddahi1", 4), ("BontempsMeddahi2", 6)):
        expected = sum(
            sum(special.eval_hermitenorm(k, z)) ** 2 / math.factorial(k)
            for k in range(3, degree + 1)
        ) / len(x)
        assert value(name, x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("x", SAMPLES)
def test_robust_scale_statistics_and_spacing_statistics(x):
    n = len(x)
    centered = x - x.mean()
    robust_scale = np.sqrt(np.pi / 2) * np.mean(abs(x - np.median(x)))
    assert value("GMG", x) == pytest.approx(x.std() / robust_scale)
    rjb = n / 6 * (np.mean(centered**3) / robust_scale**3) ** 2
    rjb += n / 64 * (np.mean(centered**4) / robust_scale**4 - 3) ** 2
    assert value("RobustJarqueBera", x) == pytest.approx(rjb)
    omega = 13.29 * np.log(x.std() / np.mean(abs(centered)))
    assert value("BonettSeier", x) == pytest.approx(np.sqrt(n + 2) * (omega - 3) / 3.54)
    # A pairwise expression for the rank-weighted Gini contrast.
    pairwise = sum(abs(a - b) for j, a in enumerate(x) for b in x[:j])
    d = pairwise / (2 * n * n * x.std())
    assert value("Dagostino", x) == pytest.approx(np.sqrt(n) * (d - 0.28209479) / 0.02998598)
    scores = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    spacing = np.diff(np.sort(x)) / np.diff(scores)
    assert value("ChenShapiro", x) == pytest.approx(
        np.sqrt(n) * (1 - spacing.mean() / x.std(ddof=1))
    )
    u = np.ptp(x) / x.std(ddof=1)
    g = np.mean(abs(centered)) / x.std()
    c = math.factorial(n) ** (1 / (n - 1)) / (2 * n)
    expected = ((c * u) ** (-(n - 1)) + g ** (-(n - 1))) ** (1 / (n - 1))
    assert value("Spiegelhalter", x) == pytest.approx(expected)


@pytest.mark.parametrize("x", SAMPLES)
def test_glb_implemented_functional_via_binomial_order_probabilities(x):
    z = np.sort((x - x.mean()) / x.std(ddof=1))
    n = len(x)
    u = stats.norm.cdf(z)
    # I_u(i,n+1-i) = P(Binomial(n,u) >= i), not a production beta call.
    transformed = sorted(stats.binom.sf(i - 1, n, p) for i, p in enumerate(u, 1))
    expected = (
        -n
        - sum(
            (2 * n + 1 - 2 * i) * np.log(p) + (2 * i - 1) * np.log1p(-p)
            for i, p in enumerate(transformed, 1)
        )
        / n
    )
    assert value("GlenLeemisBarr", x) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("x", SAMPLES)
def test_swrg_via_inverse_brownian_bridge_covariance(x):
    n = len(x)
    p = np.arange(1, n + 1) / (n + 1)
    scores = stats.norm.ppf(p)
    density = stats.norm.pdf(scores)
    covariance = (np.minimum.outer(p, p) - np.outer(p, p)) / np.outer(density, density)
    weights = np.linalg.solve(covariance, scores)
    weights /= np.linalg.norm(weights)
    expected = np.dot(weights, np.sort(x)) ** 2 / sum((x - x.mean()) ** 2)
    assert value("SWRG", x) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("n", [4, 5, 12, 25])
def test_coin_scores_against_order_statistic_quadrature_and_ols(n):
    x = np.sort(np.random.default_rng(n).normal(size=n))

    # Independent E[Z_(i)] via the order statistic density.
    def expected_score(i):
        coefficient = math.factorial(n) / (math.factorial(i - 1) * math.factorial(n - i))
        return integrate.quad(
            lambda z: (
                z
                * coefficient
                * stats.norm.pdf(z)
                * stats.norm.cdf(z) ** (i - 1)
                * stats.norm.sf(z) ** (n - i)
            ),
            -10,
            10,
            epsabs=1e-11,
        )[0]

    scores = np.array([expected_score(i) for i in range(1, n + 1)])
    design = np.column_stack([scores, scores**3])
    beta = np.linalg.lstsq(design, (x - x.mean()) / x.std(ddof=1), rcond=None)[0]
    assert value("Coin", x) == pytest.approx(beta[1] ** 2, rel=3e-3, abs=1e-6)


def test_doornik_hansen_published_appendix_transformations():
    # Doornik & Hansen author manuscript, appendix p.7; rounded printed values.
    statistic = normal.DoornikHansenNormalityGofStatistic(
        parameters_for(normal.DoornikHansenNormalityGofStatistic)
    )
    for skew, kurt, z1, z2 in zip(
        [0.19965, 0.17132, 0.15837, 1.1610],
        [2.8221, 4.1994, 3.9722, 4.5793],
        [0.63839, 0.54876, 0.50762, 3.1862],
        [0.24687, 2.5423, 2.2381, -1.3278],
        strict=True,
    ):
        assert statistic.skewness_to_z1(skew, 50) == pytest.approx(z1, abs=2e-4)
        assert statistic.kurtosis_to_z2(skew, kurt, 50) == pytest.approx(z2, abs=2e-4)


@pytest.mark.parametrize(
    "name", [name for name in normal.__all__ if not name.startswith("Abstract")]
)
def test_masked_observations_are_rejected(name):
    cls = getattr(normal, name)
    sample = np.ma.array(SAMPLES[0], mask=[True] + [False] * 23)
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(sample)
