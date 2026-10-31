"""Independent integral, multinomial and composite-null Pareto references."""

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest
from scipy import integrate, stats

from pysatl_criterion.distribution import ParetoDistributionDescriptor as Pareto
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins, ExplicitBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import pareto
from pysatl_criterion.statistics.goodness_of_fit.continuous.pareto import pareto_statistics
from tests.parameter_cases import parameters_for


@pytest.mark.parametrize("shape,scale", [(0.2, 0.01), (1, 1), (2.3, 7), (80, 1e100)])
def test_all_edf_against_independent_integrals_and_formulas(shape, scale):
    x = stats.pareto.ppf([0.08, 0.19, 0.32, 0.51, 0.68, 0.87], shape, scale=scale)
    u = stats.pareto.cdf(x, shape, scale=scale)
    n = len(x)
    i = np.arange(1, n + 1)
    dp = stats.kstest(x, stats.pareto(shape, scale=scale).cdf, alternative="greater").statistic
    dm = stats.kstest(x, stats.pareto(shape, scale=scale).cdf, alternative="less").statistic

    def discrepancy(t):
        return np.count_nonzero(u <= t) / n - t

    def integral(f):
        return integrate.quad(f, 0, 1, points=u, epsabs=1e-11)[0]

    mean = integral(discrepancy)
    r = i - 0.5
    references = {
        "KolmogorovSmirnovDPlus": dp,
        "KolmogorovSmirnovDMinus": dm,
        "KolmogorovSmirnov": max(dp, dm),
        "AndersonDarling": n * integral(lambda t: discrepancy(t) ** 2 / (t * (1 - t))),
        "CramerVonMises": n * integral(lambda t: discrepancy(t) ** 2),
        "Watson": n * integral(lambda t: (discrepancy(t) - mean) ** 2),
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
    assert {name + "ParetoGofStatistic" for name in references} == set(
        pareto_statistics._EDF_STATISTICS
    )
    for prefix, expected in references.items():
        cls = getattr(pareto, prefix + "ParetoGofStatistic")
        criterion = cls(Pareto.DEFAULT.parse({"shape": shape, "scale": scale}))
        assert criterion.execute_statistic(x[::-1]) == pytest.approx(expected, rel=2e-10, abs=1e-10)


@pytest.mark.parametrize(
    "prefix", ["AndersonDarling", "LiaoShimokawa", "ZhangK", "ZhangA", "ZhangC"]
)
def test_logarithmic_kernels_against_decimal_in_extreme_tail(prefix):
    x = [2.0, 1e20, 1e100]
    with localcontext() as context:
        context.prec = 240
        n = Decimal(len(x))
        u = [1 - Decimal.from_float(v) ** -2 for v in x]
        logs = [v.ln() for v in u]
        survival = [(1 - v).ln() for v in u]
        ranks = [Decimal(i) - Decimal("0.5") for i in range(1, 4)]
        expected = {
            "AndersonDarling": -n
            - sum((2 * Decimal(i) - 1) * (logs[i - 1] + survival[-i]) for i in range(1, 4)) / n,
            "LiaoShimokawa": sum(
                max(Decimal(i) / n - v, v - Decimal(i - 1) / n) / (v * (1 - v)).sqrt()
                for i, v in enumerate(u, 1)
            )
            / n.sqrt(),
            "ZhangK": max(
                r * ((r / n).ln() - a) + (n - r) * (((n - r) / n).ln() - b)
                for r, a, b in zip(ranks, logs, survival, strict=True)
            ),
            "ZhangA": -sum(
                a / (n - r) + b / r for r, a, b in zip(ranks, logs, survival, strict=True)
            ),
            "ZhangC": sum(
                (b - a - ((n - Decimal(i) + Decimal("0.25")) / (Decimal(i) - Decimal("0.75"))).ln())
                ** 2
                for i, (a, b) in enumerate(zip(logs, survival, strict=True), 1)
            ),
        }[prefix]
    cls = getattr(pareto, prefix + "ParetoGofStatistic")
    criterion = cls(Pareto.DEFAULT.parse({"shape": 2, "scale": 1}))
    assert criterion.execute_statistic(x) == pytest.approx(float(expected), rel=2e-12)
    assert criterion.execute_statistic([1, 2, 3]) == np.inf


FREQUENCIES = {
    "Chi2Pearson": 1,
    "FreemanTukey": -0.5,
    "ModifiedLogLikelihood": -1,
    "NeymanChi2": -2,
    "CressieRead": 2 / 3,
}


@pytest.mark.parametrize("prefix,power", FREQUENCIES.items())
@pytest.mark.parametrize("counts", [(3, 4, 5), (0, 7, 5), (12, 0, 0)])
def test_frequency_against_scipy_power_divergence(prefix, power, counts):
    # Shape=2, scale=3: the three cells have probabilities 1/2, 1/4, 1/4.
    edges = (3, 3 * np.sqrt(2), 6, np.inf)
    sample = np.repeat([3.2, 5.0, 8.0], counts)
    expected = len(sample) * np.array([0.5, 0.25, 0.25])
    if 0 in counts and power <= -1:
        reference = np.inf
    elif power == -0.5:
        # SciPy's generic formula has 0*inf at a zero count; take the continuous limit.
        reference = 4 * sum((np.sqrt(counts) - np.sqrt(expected)) ** 2)
    else:
        reference = stats.power_divergence(counts, expected, lambda_=power).statistic
    cls = getattr(pareto, prefix + "ParetoGofStatistic")
    criterion = cls(Pareto.DEFAULT.parse({"shape": 2, "scale": 3}), binning=ExplicitBins(edges))
    assert criterion.execute_statistic(sample) == pytest.approx(reference)
    # Equal probability binning must give the same multinomial calculation for its own cells.
    criterion = cls(Pareto.DEFAULT.parse({"shape": 2, "scale": 3}), binning=EqualProbabilityBins(3))
    sample = stats.pareto.ppf(np.repeat([0.1, 0.5, 0.9], [3, 4, 5]), 2, scale=3)
    assert criterion.execute_statistic(sample) == pytest.approx(
        stats.power_divergence([3, 4, 5], [4, 4, 4], lambda_=power).statistic
    )


@pytest.mark.parametrize("n", [3, 7, 20])
def test_greenwood_exact_dirichlet_moments(n):
    rng = np.random.default_rng(8128 + n)
    samples = 2 * np.exp(rng.exponential(size=(6000, n)) / 3)
    criterion = pareto.GreenwoodParetoGofStatistic(Pareto.DEFAULT.parse({}))
    values = np.array([criterion.execute_statistic(x) for x in samples])
    mean = 2 / n
    variance = 4 * (n - 2) / (n * n * (n + 1) * (n + 2))
    assert abs(values.mean() - mean) < 6 * np.sqrt(variance / len(values))
    assert values.var() == pytest.approx(variance, rel=0.12)
    assert np.all(values >= 1 / (n - 1))
    assert np.all(values <= 1)


def test_obradovic_exact_reference_and_power_invariance():
    criterion = pareto.ObradovicParetoGofStatistic(Pareto.DEFAULT.parse({"scale": 3}))
    z = [Fraction(x) for x in [1, 2, 3, 4, 8, 16]]
    pairs = [max(a / b, b / a) for a, b in combinations(z, 2)]
    reference = sum(
        Fraction(sum(r <= t for r in pairs), len(pairs)) - Fraction(sum(v <= t for v in z), len(z))
        for t in z
    ) / len(z)
    for power in [1, 2, 3]:
        x = [3 * float(v**power) for v in z]
        assert criterion.execute_statistic(x) == pytest.approx(float(reference))


@pytest.mark.parametrize("m", [1, 2, 3])
def test_kl_scipy_entropy_and_mle(m):
    x = np.array([1.1, 1.3, 1.8, 2.4, 4.0, 8.0, 17.0, 50.0])
    fitted_shape, _, fitted_scale = stats.pareto.fit(x, floc=0)
    reference = -stats.differential_entropy(x, window_length=m, method="vasicek")
    reference -= np.mean(stats.pareto.logpdf(x, fitted_shape, scale=fitted_scale))
    criterion = pareto.LequesneKlParetoGofStatistic(Pareto.DEFAULT.parse({}))
    assert criterion.execute_statistic(x, m=m) == pytest.approx(reference)


@pytest.mark.parametrize("name", [n for n in pareto.__all__ if not n.startswith("Abstract")])
def test_masked_observations_are_not_silently_used(name):
    cls = getattr(pareto, name)
    x = np.ma.array([1.1, 1.4, 1.9, 2.7, 4.2, 8.0], mask=[False] * 5 + [True])
    with pytest.raises(ValueError, match="masked"):
        cls(parameters_for(cls)).execute_statistic(x)
