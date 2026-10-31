"""Independent formula references and log-transform composition checks for all 55 statistics."""

from decimal import Decimal, localcontext
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import LogNormalDistributionDescriptor as Distribution
from pysatl_criterion.distribution import NormalDistributionDescriptor as Normal
from pysatl_criterion.statistics.goodness_of_fit.binning import ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import log_normal as ln
from pysatl_criterion.statistics.goodness_of_fit.continuous import normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import (
    log_normal_statistics,
)
from pysatl_criterion.utils.generator import get_hypothesis_generator
from tests.parameter_cases import parameters_for


CLASSES = [getattr(ln, name) for name in ln.__all__ if not name.startswith("Abstract")]
WRAPPERS = [
    cls
    for cls in CLASSES
    if cls.__name__.replace("LogNormal", "Normality") in normal.__all__
    and cls.__name__ not in log_normal_statistics._EDF_STATISTICS
    and cls.__name__ not in log_normal_statistics._FREQUENCY_STATISTICS
]
LOGS = [
    np.random.default_rng(18).normal(size=24),
    np.random.default_rng(19).uniform(-2, 2, size=24),
    np.array([-2, -1, -1, -0.5, 0, 0, 0.4, 0.4, 0.8, 1, 1.5, 2.5]),
]


@pytest.mark.parametrize("cls", WRAPPERS)
@pytest.mark.parametrize("y", LOGS)
def test_logarithmic_composition_and_power_scale_invariance(cls, y):
    criterion = cls(Distribution.DEFAULT.parse({}))
    reference = getattr(normal, cls.__name__.replace("LogNormal", "Normality"))(
        Normal.DEFAULT.parse({})
    )
    expected = reference.execute_statistic(y)
    assert criterion.alternative().type() == reference.alternative().type()
    for a, b in [(0, 1), (200, 0.7), (-300, 1.3)]:
        sample = np.exp(a + b * y)
        original = sample.copy()
        assert criterion.execute_statistic(sample) == pytest.approx(expected, rel=3e-8, abs=2e-9)
        np.testing.assert_array_equal(sample, original)
    assert criterion.hypothesis().parameters() == {}
    assert get_hypothesis_generator(criterion).parameters() == {"mu": 0, "s": 1}


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[1, 2], [3, 4]],
        [1, np.nan],
        [1, np.inf],
        [1 + 2j, 3],
        np.ma.array([1, 2, 3], mask=[0, 1, 0]),
    ],
)
def test_invalid_data_cannot_silently_enter_a_statistic(cls, bad):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(bad)


@pytest.mark.parametrize("name", log_normal_statistics._EDF_STATISTICS)
@pytest.mark.parametrize("y", LOGS[:2])
def test_edf_against_independent_integrals_and_rank_sums(name, y):
    criterion = getattr(ln, name)(Distribution.DEFAULT.parse({"mu": 0.4, "s": 1.2}))
    z = (np.sort(y) - 0.4) / 1.2
    # Independent density quadrature, not a production CDF or EDF kernel.
    u = np.array(
        [
            integrate.quad(lambda v: np.exp(-v * v / 2) / np.sqrt(2 * np.pi), -np.inf, t)[0]
            for t in z
        ]
    )
    n = len(z)
    i = np.arange(1, n + 1)
    plus, minus = np.max(i / n - u), np.max(u - (i - 1) / n)
    code = criterion.short_code()
    if code in {"KS", "KS_PLUS", "KS_MINUS"}:
        expected = {"KS": max(plus, minus), "KS_PLUS": plus, "KS_MINUS": minus}[code]
    elif code in {"AD", "CVM", "WATSON"}:
        center = 0.5 - u.mean() if code == "WATSON" else 0
        expected = 0
        for j, (lo, hi) in enumerate(pairwise(np.r_[0, u, 1])):

            def f(t, j=j):
                v = (j / n - t - center) ** 2
                return v / (t * (1 - t)) if code == "AD" else v

            expected += n * integrate.quad(f, lo, hi, epsabs=1e-12)[0]
    elif code.startswith("KUIPER"):
        factor = {
            "KUIPER": 1,
            "KUIPER_SCALED": np.sqrt(n),
            "KUIPER_CORRECTED": np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n),
        }[code]
        expected = factor * (plus + minus)
    elif code == "LIAO_SHIMOKAWA":
        expected = np.sum(np.maximum(i / n - u, u - (i - 1) / n) / np.sqrt(n * u * (1 - u)))
    elif code in {"ZHANG_ZA", "ZWA"}:
        expected = -np.sum(np.log(u) / (n - i + 0.5) + np.log1p(-u) / (i - 0.5))
        if code == "ZWA":
            expected = 10 * expected - 32
    elif code in {"ZHANG_ZC", "ZWC"}:
        expected = np.sum(np.log((1 - u) / u * (i - 0.75) / (n - i + 0.25)) ** 2)
    elif code == "ZHANG_ZK":
        expected = np.max(
            (i - 0.5) * np.log((i - 0.5) / (n * u))
            + (n - i + 0.5) * np.log((n - i + 0.5) / (n * (1 - u)))
        )
    else:
        pytest.fail(f"Missing definition for {code}")
    assert criterion.execute_statistic(np.exp(y)) == pytest.approx(expected, rel=2e-9, abs=2e-10)


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
def test_frequency_against_multinomial_power_divergence(name, lambda_):
    sample = np.exp(LOGS[0])
    edges = [0, 0.7, 1.5, np.inf]
    p = np.diff(stats.lognorm(s=0.8, scale=1.1).cdf(edges))
    observed = np.histogram(sample, bins=edges)[0]
    expected = stats.power_divergence(observed, len(sample) * p, lambda_=lambda_).statistic
    criterion = getattr(ln, name + "LogNormalGofStatistic")(
        Distribution.SHAPE_SCALE.parse({"s": 0.8, "scale": 1.1}), binning=ExplicitBins(edges)
    )
    assert criterion.execute_statistic(sample) == pytest.approx(expected)


def test_quesenberry_miller_has_all_spacings_and_no_wraparound():
    u = np.array([0.1, 0.3, 0.6, 0.8])
    sample = stats.lognorm(s=0.7, scale=1.2).ppf(u)
    criterion = ln.QuesenberryMillerLogNormalGofStatistic(
        Distribution.SHAPE_SCALE.parse({"s": 0.7, "scale": 1.2})
    )
    # Spacings .1,.2,.3,.2,.2; squares .22, adjacent products .18.
    assert criterion.execute_statistic(sample) == pytest.approx(0.40)


def decimal_ratio(z, r):
    """70-digit direct expression, independent of production cumulant series."""
    with localcontext() as context:
        context.prec = 70
        values = [Decimal(str(x)) for x in z]
        r = Decimal(str(r))
        n = Decimal(len(values))
        exponentials = [(r * x).exp() for x in values]
        mean = sum(exponentials) / n
        d = (
            2 * mean.ln()
            - r * sum(values) / n
            - r * sum(x * e for x, e in zip(values, exponentials, strict=True)) / sum(exponentials)
        )
        t = r * r
        variance = t.exp() * (t * t - 3 * t + 4) - 4 - t
        return float(d / variance.sqrt())


@pytest.mark.parametrize("r", [-5, -1, -0.1, -0.001, -1e-6, 1e-6, 0.001, 0.1, 1, 5])
def test_kl_ratio_matches_high_precision_reference(r):
    z = ln.AbstractLogNormalGofStatistic._standardized_logs(np.exp(LOGS[0]))
    assert ln.AbstractLogNormalGofStatistic._kl_ratio_function(z)(r) == pytest.approx(
        decimal_ratio(z, r), rel=2e-7, abs=2e-9
    )


def test_kl_zero_limit_and_integral_are_finite_and_correct():
    x = np.exp(LOGS[0])
    z = (np.log(x) - np.log(x).mean()) / np.log(x).std()
    limit = -np.mean(z**3) / np.sqrt(6)
    assert ln.AbstractLogNormalGofStatistic._kl_ratio_function(z)(0) == pytest.approx(limit)
    # Independent quadrature of the high-precision direct expression (no production series).
    expected = (
        len(z)
        * integrate.quad(lambda r: decimal_ratio(z, r) ** 2, -5, 5, points=[0], epsabs=1e-8)[0]
    )
    criterion = ln.KLIntegralLogNormalGoFStatistic(Distribution.DEFAULT.parse({}))
    assert criterion.execute_statistic(x) == pytest.approx(expected, rel=2e-8)


@pytest.mark.parametrize(
    "cls", [ln.KLIntegralLogNormalGoFStatistic, ln.KLSupremumLogNormalGoFStatistic]
)
def test_kl_scale_invariance_domains_and_calibration(cls):
    criterion = cls(Distribution.DEFAULT.parse({}))
    expected = criterion.execute_statistic(np.exp(LOGS[0]))
    for shift in [-600, 600]:
        assert criterion.execute_statistic(np.exp(shift + 0.6 * LOGS[0])) == pytest.approx(
            expected, rel=2e-7
        )
    for bad in [[1], [1, 1], [0, 1, 2], [-1, 1, 2]]:
        with pytest.raises(ValueError):
            criterion.execute_statistic(bad)
    with pytest.raises(NotImplementedError, match="Monte Carlo"):
        criterion.calculate_critical_value(50, 0.025)
    assert get_hypothesis_generator(criterion).parameters() == {"mu": 0, "s": 1}


def test_kl_grid_maximum_against_high_precision_direct_expression():
    sample = np.exp(LOGS[0])
    z = (np.log(sample) - np.log(sample).mean()) / np.log(sample).std()
    expected = np.sqrt(len(z)) * max(abs(decimal_ratio(z, r)) for r in np.linspace(-5, 5, 1000))
    criterion = ln.KLSupremumLogNormalGoFStatistic(Distribution.DEFAULT.parse({}))
    assert criterion.execute_statistic(sample) == pytest.approx(expected, rel=2e-7)


def test_legacy_critical_value_never_returns_a_negative_threshold():
    criterion = ln.KLIntegralLogNormalGoFStatistic(Distribution.DEFAULT.parse({}))
    with pytest.raises(ValueError, match="Monte Carlo"):
        criterion.calculate_critical_value(2, 0.05)
