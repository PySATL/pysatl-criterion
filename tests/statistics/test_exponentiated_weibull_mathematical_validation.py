"""Independent formulas, integrals and decimal references for the specified null."""

import re
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import ExponentiatedWeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.alternative import AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import exponentiated_weibull as ew
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    exponentiated_weibull_statistics as implementation,
)


EDF = implementation._EDF_STATISTICS
FREQUENCY = implementation._FREQUENCY_STATISTICS
LOG_PREFIXES = ["AndersonDarling", "LiaoShimokawa", "ZhangK", "ZhangA", "ZhangC"]


def criterion(prefix, a=1, k=1, scale=1, **kwargs):
    return getattr(ew, prefix + "ExponentiatedWeibullGofStatistic")(
        Distribution.DEFAULT.parse({"exponent": a, "shape": k, "scale": scale}), **kwargs
    )


@pytest.mark.parametrize("a,k,scale", [(0.3, 0.7, 0.02), (1, 1, 1), (2.3, 4, 8), (80, 0.2, 1e100)])
def test_all_edf_against_integrals_and_independent_formulas(a, k, scale):
    dist = stats.exponweib(a, k, scale=scale)
    x = dist.ppf([0.04, 0.19, 0.32, 0.51, 0.68, 0.87, 0.96])
    u = dist.cdf(x)
    n = len(x)
    i = np.arange(1, n + 1)
    r = i - 0.5
    dp = stats.kstest(x, dist.cdf, alternative="greater").statistic
    dm = stats.kstest(x, dist.cdf, alternative="less").statistic

    def discrepancy(t):
        return np.count_nonzero(u <= t) / n - t

    def integral(f):
        return integrate.quad(f, 0, 1, points=u, epsabs=1e-11)[0]

    center = integral(discrepancy)
    references = {
        "KolmogorovSmirnovDPlus": dp,
        "KolmogorovSmirnovDMinus": dm,
        "KolmogorovSmirnov": max(dp, dm),
        "AndersonDarling": n * integral(lambda t: discrepancy(t) ** 2 / (t * (1 - t))),
        "CrammerVonMises": n * integral(lambda t: discrepancy(t) ** 2),
        "Watson": n * integral(lambda t: (discrepancy(t) - center) ** 2),
        "Kuiper": dp + dm,
        "KuiperScaled": np.sqrt(n) * (dp + dm),
        "KuiperCorrected": (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * (dp + dm),
        "LiaoShimokawa": sum(
            max(j / n - v, v - (j - 1) / n) / np.sqrt(v * (1 - v))
            for j, v in zip(i, u, strict=True)
        )
        / np.sqrt(n),
        "ZhangK": max(r * np.log(r / (n * u)) + (n - r) * np.log((n - r) / (n * (1 - u)))),
        "ZhangA": -sum(np.log(u) / (n - r) + np.log1p(-u) / r),
        "ZhangC": sum((np.log((1 - u) / u) - np.log((n - i + 0.25) / (i - 0.75))) ** 2),
    }
    assert {p + "ExponentiatedWeibullGofStatistic" for p in references} == set(EDF)
    for prefix, expected in references.items():
        value = criterion(prefix, a, k, scale).execute_statistic(x[::-1])
        assert value == pytest.approx(expected, rel=2e-10, abs=1e-10)


@pytest.mark.parametrize(
    "a,x",
    [(2.3, [0.01, 1.0, 40.0, 100.0]), (1e-320, [1.0, 2.0, 10.0]), (1e200, [460.0, 461.0, 462.0])],
)
def test_log_statistics_against_800_digit_decimal(a, x):
    with localcontext() as context:
        context.prec = 800
        exponent = Decimal.from_float(a)
        n = Decimal(len(x))
        u = [((1 - (-Decimal.from_float(v)).exp()).ln() * exponent).exp() for v in x]
        lc = [v.ln() for v in u]
        ls = [(1 - v).ln() for v in u]
        r = [Decimal(j) - Decimal("0.5") for j in range(1, len(x) + 1)]
        references = {
            "AndersonDarling": -n
            - sum((2 * Decimal(j) - 1) * (lc[j - 1] + ls[-j]) for j in range(1, len(x) + 1)) / n,
            "LiaoShimokawa": sum(
                max(Decimal(j) / n - v, v - Decimal(j - 1) / n) / (v * (1 - v)).sqrt()
                for j, v in enumerate(u, 1)
            )
            / n.sqrt(),
            "ZhangK": max(
                t * ((t / n).ln() - log_cdf) + (n - t) * (((n - t) / n).ln() - s)
                for t, log_cdf, s in zip(r, lc, ls, strict=True)
            ),
            "ZhangA": -sum(
                log_cdf / (n - t) + s / t for t, log_cdf, s in zip(r, lc, ls, strict=True)
            ),
            "ZhangC": sum(
                (
                    s
                    - log_cdf
                    - ((n - Decimal(j) + Decimal(".25")) / (Decimal(j) - Decimal(".75"))).ln()
                )
                ** 2
                for j, (log_cdf, s) in enumerate(zip(lc, ls, strict=True), 1)
            ),
        }
    for prefix, expected in references.items():
        assert criterion(prefix, a).execute_statistic(x) == pytest.approx(
            float(expected), rel=2e-11
        )


@pytest.mark.parametrize("prefix", LOG_PREFIXES)
def test_exact_zero_has_infinite_logarithmic_penalty(prefix):
    assert np.isposinf(criterion(prefix, 2.3, 1.7).execute_statistic([0.0, 1.0, 2.0]))


@pytest.mark.parametrize("prefix", ["KolmogorovSmirnov", "CrammerVonMises", "Watson"])
def test_single_observation_and_ties(prefix):
    for x in ([0.0], [1.0], [1.0, 1.0, 1.0]):
        u = stats.exponweib(1, 1).cdf(x)
        expected = {
            "KolmogorovSmirnov": max(u[0], 1 - u[0]),
            "CrammerVonMises": len(x) * (u[0] ** 3 + (1 - u[0]) ** 3) / 3,
            "Watson": 1 / (12 * len(x)) + np.sum(((np.arange(len(x)) + 0.5) / len(x) - 0.5) ** 2),
        }
        assert criterion(prefix).execute_statistic(x) == pytest.approx(expected[prefix])


@pytest.mark.parametrize(
    "direction,prefix",
    [
        (AlternativeType.RIGHT, "KolmogorovSmirnovDPlus"),
        (AlternativeType.LEFT, "KolmogorovSmirnovDMinus"),
        (AlternativeType.TWO_TAILED, "KolmogorovSmirnov"),
    ],
)
def test_legacy_ks_direction_is_a_cdf_direction(direction, prefix):
    x = [0.1, 0.4, 1, 2.0]
    value = criterion("KolmogorovSmirnov", alternative_type=direction, mode="asymp")
    assert isinstance(value.alternative(), RightAlternative)
    assert value.execute_statistic(x) == criterion(prefix).execute_statistic(x)


@pytest.mark.parametrize(
    "prefix,power",
    [
        ("Chi2Pearson", 1),
        ("FreemanTukey", -0.5),
        ("ModifiedLogLikelihood", -1),
        ("NeymanChi2", -2),
        ("CressieRead", 2 / 3),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_frequency_against_scipy_multinomial_divergence(prefix, power, explicit):
    a, k, scale = 2.3, 1.7, 4.0
    dist = stats.exponweib(a, k, scale=scale)
    x = dist.ppf([0.02, 0.08, 0.12, 0.29, 0.39, 0.51, 0.61, 0.69, 0.78, 0.89, 0.99])
    if explicit:
        edges = [0, *dist.ppf([0.15, 0.45, 0.75]), np.inf]
        binning = ExplicitBins(tuple(edges))
        observed = np.histogram(x, edges)[0]
        expected = len(x) * np.diff(dist.cdf(edges))
    else:
        binning = EqualProbabilityBins(count=4)
        observed = np.histogram(dist.cdf(x), np.linspace(0, 1, 5))[0]
        expected = np.full(4, len(x) / 4)
    reference = stats.power_divergence(observed, expected, lambda_=power).statistic
    assert criterion(prefix, a, k, scale, binning=binning).execute_statistic(x) == pytest.approx(
        reference
    )


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("Chi2Pearson", 4),
        ("FreemanTukey", 32 - 16 * np.sqrt(2)),
        ("ModifiedLogLikelihood", np.inf),
        ("NeymanChi2", np.inf),
        ("CressieRead", 1.8 * 4 * (2 ** (2 / 3) - 1)),
    ],
)
def test_empty_bins_and_default_singleton(prefix, expected):
    value = criterion(prefix, binning=EqualProbabilityBins(count=2))
    assert value.execute_statistic([0.0, 0.0, 0.0, 0.0]) == pytest.approx(expected)
    # The default has TWO bins even at n=1, unlike the historical sqrt(n) rule.
    assert criterion(prefix).execute_statistic([0.0]) == pytest.approx(expected / 4)


def test_default_bin_count_matches_documented_rule():
    x = np.linspace(0.1, 4, 27)
    bins = int(np.ceil(2 * len(x) ** 0.4))
    observed = np.histogram(stats.exponweib(1, 1).cdf(x), np.linspace(0, 1, bins + 1))[0]
    expected = np.full(bins, len(x) / bins)
    assert criterion("Chi2Pearson").execute_statistic(x) == pytest.approx(
        stats.chisquare(observed, expected).statistic
    )


def test_every_criterion_has_documentation_reference_and_working_example():
    root = Path(__file__).resolve().parents[2]
    folder = root / "docs/criteria/goodness-of-fit/exponentiated-weibull"
    index = (folder / "index.md").read_text()
    navigation = (root / "mkdocs.yml").read_text()
    pages = list(folder.glob("*_criterion.md"))
    assert len(pages) == len(EDF) + len(FREQUENCY) == 18
    for name in (*EDF, *FREQUENCY):
        matching = [p for p in pages if f"`{name}`" in p.read_text()]
        assert len(matching) == 1, name
        page = matching[0]
        content = page.read_text()
        assert page.name in index and page.name in navigation
        assert "## Formula" in content and "## References" in content
        assert "https://doi.org/" in content
        examples = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
        assert examples
        for code in examples:
            namespace = {}
            exec(compile(code, str(page), "exec"), namespace)  # noqa: S102
            assert np.isfinite(namespace["value"])
        doc = getattr(ew, name).__doc__
        assert "https://" in doc and "Examples" in doc


def test_shape_exponent_product_survives_intermediate_overflow():
    x = np.array([0.01, 0.1, 0.4, 0.8])
    value = criterion("KolmogorovSmirnov", 1e-308, 1e308)
    expected = stats.kstest(x, "uniform").statistic
    assert value.execute_statistic(x) == pytest.approx(expected)
    value = criterion("Chi2Pearson", 1e-308, 1e308, binning=EqualProbabilityBins(count=2))
    assert value.execute_statistic(x) == pytest.approx(1.0)


def test_frequency_cdf_avoids_underflow_before_exponentiation():
    # (x**k)**a is approximately x here. SciPy's intermediate x**k is zero.
    x = np.array([0.01, 0.1, 0.4, 0.8])
    value = criterion("Chi2Pearson", 0.001, 1000.0, binning=EqualProbabilityBins(count=2))
    assert value.execute_statistic(x) == pytest.approx(1.0)


def test_bound_distribution_support_and_shape_contract():
    value = criterion("KolmogorovSmirnov", 2.3, 1.7, 4.0)
    bound = value._bind_distribution(None)
    dist = stats.exponweib(2.3, 1.7, scale=4.0)
    x = np.array([[-np.inf, -1.0, 0.0], [1.0, 4.0, np.inf]])
    for method in ["cdf", "sf", "logcdf", "logsf"]:
        actual = getattr(bound, method)(x)
        np.testing.assert_allclose(actual, getattr(dist, method)(x), rtol=1e-13)
        assert np.ndim(getattr(bound, method)(1.0)) == 0
        assert np.isnan(getattr(bound, method)(np.nan))


def test_public_docstring_examples():
    import doctest

    result = doctest.testmod(ew, extraglobs={"Distribution": Distribution, "np": np})
    assert result.attempted == 4 * (len(EDF) + len(FREQUENCY))
    assert result.failed == 0
