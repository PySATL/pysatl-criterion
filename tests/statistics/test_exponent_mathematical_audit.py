"""Independent definitions for every exponential class and audit regressions."""

import inspect
import itertools
import math

import networkx as nx
import numpy as np
import pytest
from scipy import integrate, special, stats

from pysatl_criterion.distribution import ExponentialDistributionDescriptor as E
from pysatl_criterion.hypothesis_testing.exponential_monte_carlo import monte_carlo_exponential
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import exponent as exp


CLASSES = {
    name.removesuffix("ExponentialityGofStatistic"): getattr(exp, name)
    for name in exp.__all__
    if not inspect.isabstract(getattr(exp, name))
}


def _graph_references(x):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(x)))
    graph.add_edges_from(
        (i, j)
        for i, j in itertools.combinations(range(len(x)), 2)
        if abs(x[i] - x[j]) < (max(x) - min(x)) / 10
    )
    return {
        "GraphEdgesNumber": graph.number_of_edges(),
        "GraphMaxDegree": max(dict(graph.degree).values()),
        "GraphAverageDegree": 2 * graph.number_of_edges() / len(x),
        "GraphConnectedComponents": nx.number_connected_components(graph),
        "GraphCliqueNumber": max(map(len, nx.find_cliques(graph))),
        "GraphIndependenceNumber": max(map(len, nx.find_cliques(nx.complement(graph)))),
    }


def _edf_references(x, rate):
    n = len(x)
    u = stats.expon.cdf(sorted(x), scale=1 / rate)
    i = np.arange(1, n + 1)
    plus = stats.kstest(x, stats.expon(scale=1 / rate).cdf, alternative="greater").statistic
    minus = stats.kstest(x, stats.expon(scale=1 / rate).cdf, alternative="less").statistic
    cvm = stats.cramervonmises(x, stats.expon(scale=1 / rate).cdf).statistic
    return {
        "KolmogorovSmirnovDPlus": plus,
        "KolmogorovSmirnovDMinus": minus,
        "KolmogorovSmirnov": max(plus, minus),
        "Lilliefors": max(plus, minus),
        "AndersonDarling": -n
        - sum(
            (2 * j - 1) * math.log(v) + (2 * (n - j) + 1) * math.log1p(-v)
            for j, v in zip(i, u, strict=True)
        )
        / n,
        "CramerVonMises": cvm,
        "LiaoShimokawa": sum(
            max(j / n - v, v - (j - 1) / n) / math.sqrt(v * (1 - v))
            for j, v in zip(i, u, strict=True)
        )
        / math.sqrt(n),
        "Watson": cvm - n * (np.mean(u) - 0.5) ** 2,
        "Kuiper": plus + minus,
        "KuiperScaled": math.sqrt(n) * (plus + minus),
        "KuiperCorrected": (math.sqrt(n) + 0.155 + 0.24 / math.sqrt(n)) * (plus + minus),
        "ZhangK": max(
            (j - 0.5) * math.log((j - 0.5) / (n * v))
            + (n - j + 0.5) * math.log((n - j + 0.5) / (n * (1 - v)))
            for j, v in zip(i, u, strict=True)
        ),
        "ZhangA": -sum(
            math.log(v) / (n - j + 0.5) + math.log1p(-v) / (j - 0.5)
            for j, v in zip(i, u, strict=True)
        ),
        "ZhangC": sum(
            math.log((1 - v) / v * (j - 0.75) / (n - j + 0.25)) ** 2
            for j, v in zip(i, u, strict=True)
        ),
    }


def _specialized_references(x):
    n = len(x)
    mean = sum(x) / n
    y = np.array(sorted(x)) / mean
    d = [
        (n - j) * (v - prev)
        for j, (prev, v) in enumerate(zip([0, *sorted(x)[:-1]], sorted(x), strict=True))
    ]
    r1, r2 = max(1, round(n / 2)), max(1, round(n / 4))
    # Direct integrals, rather than the production pair sums and analytic moments.
    elt = integrate.quad(
        lambda t: n * np.mean((1 - y * (1 + t)) * np.exp(-t * y)) ** 2 * np.exp(-t), 0, np.inf
    )[0]
    bhc = sum(
        integrate.quad(lambda t: n * np.mean((y - t - 1) * (y > t)) ** 2 * np.exp(-t), a, b)[0]
        for a, b in itertools.pairwise([0, *y])
    )
    return {
        "EppsPulley": math.sqrt(48 * n) * (sum(math.exp(-v) for v in y) / n - 0.5),
        "Ahsanullah": sum(
            (abs(a - b) < c) - (2 * min(a, b) < c)
            for a, b, c in itertools.product(map(float, x), repeat=3)
        )
        / n**3,
        "Atkinson": math.sqrt(n)
        * abs((sum(v**0.99 for v in y) / n) ** (1 / 0.99) - special.gamma(1.99) ** (1 / 0.99)),
        "CoxOakes": n + sum((1 - v) * math.log(v) for v in y),
        "Deshpande": sum(a > 0.44 * b for a, b in itertools.permutations(x, 2)) / (n * (n - 1)),
        "Epstein": 2 * n * math.log(np.mean(d) / stats.gmean(d)) / (1 + (n + 1) / (6 * n)),
        "Frozini": sum(abs(stats.expon.cdf(v) - (j - 0.5) / n) for j, v in enumerate(y, 1))
        / math.sqrt(n),
        "Gini": sum(abs(a - b) for a, b in itertools.combinations(x, 2)) / ((n - 1) * sum(x)),
        "Gnedenko": np.mean(d[:r1]) / np.mean(d[r1:]),
        "Harris": np.mean(d[:r2] + d[-r2:]) / np.mean(d[r2:-r2]),
        "HegazyGreen1": np.mean(abs(y - stats.expon.ppf(np.arange(1, n + 1) / (n + 1)))),
        "HegazyGreen2": np.mean((y - stats.expon.ppf(np.arange(1, n + 1) / (n + 1))) ** 2),
        "HollanderProshan": sum(a > b + c for a, b, c in itertools.permutations(x, 3))
        / (n * (n - 1) * (n - 2)),
        "KimberMichael": max(
            abs(
                2
                / math.pi
                * (math.asin(math.sqrt(stats.expon.cdf(v))) - math.asin(math.sqrt((j - 0.5) / n)))
            )
            for j, v in enumerate(y, 1)
        ),
        "Kochar": math.sqrt(108 * n / 17)
        * sum(
            (2 * (1 - j / (n + 1)) * (1 - math.log1p(-j / (n + 1))) - 1) * v
            for j, v in enumerate(y, 1)
        )
        / n,
        "Lorenz": sum(sorted(x)[: math.floor(n * 0.5)]) / sum(x),
        "Moran": np.euler_gamma + math.log(math.prod(y)) / n,
        "Pietra": sum(abs(v - 1) for v in y) / (2 * n),
        "ShapiroWilk": n * (mean - min(x)) ** 2 / ((n - 1) * sum((v - mean) ** 2 for v in x)),
        "Rossberg": sum(sorted(t)[1] - min(t) < v for t in itertools.combinations(x, 3) for v in x)
        / (n * math.comb(n, 3))
        - sum(min(t) < v for t in itertools.combinations(x, 2) for v in x) / (n * math.comb(n, 2)),
        "We": sum((v - mean) ** 2 for v in x) / sum(x) ** 2,
        "WongWong": max(x) / min(x),
        "Greenwood": sum(v * v for v in x) / sum(x) ** 2,
        "BaringhausHenzeELT": elt,
        "BaringhausHenzeBHC": bhc,
        **_graph_references(x),
    }


@pytest.mark.parametrize("seed", [3, 17, 29])
def test_every_criterion_against_independent_definitions(seed):
    x = np.random.default_rng(seed).uniform(0.05, 3, 8)
    references = _specialized_references(x)
    references.update(_edf_references(x, 1 / np.mean(x)))
    powers = {
        "Chi2Pearson": 1,
        "FreemanTukey": -0.5,
        "ModifiedLogLikelihood": -1,
        "NeymanChi2": -2,
        "CressieRead": 2 / 3,
    }
    assert set(references) | set(powers) == set(CLASSES)
    for name, cls in CLASSES.items():
        for support in cls.supported_hypotheses():
            fixed = bool(support.fixed_parameters)
            rate = 0.8 if fixed else 1 / np.mean(x)
            settings = {"binning": EqualProbabilityBins(count=2)} if name in powers else {}
            obj = cls(E.DEFAULT.parse({"lam": rate} if fixed else {}), **settings)
            if name in powers:
                observed = np.histogram(x, stats.expon.ppf([0, 0.5, 1], scale=1 / rate))[0]
                expected = stats.power_divergence(observed, [4, 4], lambda_=powers[name]).statistic
            elif name + "ExponentialityGofStatistic" in exp.exponent_statistics._EDF_STATISTICS:
                expected = _edf_references(x, rate)[name]
            else:
                expected = references[name]
            assert obj.execute_statistic(x) == pytest.approx(expected, rel=1e-10, abs=1e-12), (
                name,
                fixed,
            )


def test_cox_oakes_is_profile_weibull_score():
    x = np.array([0.15, 0.3, 0.9, 1.5, 2.4, 3.0])
    n = len(x)

    def log_likelihood(shape):
        return (
            n * np.log(shape) - n * np.log(np.mean(x**shape)) + (shape - 1) * np.sum(np.log(x)) - n
        )

    h = 1e-5
    score = (log_likelihood(1 + h) - log_likelihood(1 - h)) / (2 * h)
    assert exp.CoxOakesExponentialityGofStatistic(E.DEFAULT.parse({})).execute_statistic(
        x
    ) == pytest.approx(score, abs=1e-8)


@pytest.mark.parametrize(
    "cls",
    [exp.NeymanChi2ExponentialityGofStatistic, exp.ModifiedLogLikelihoodExponentialityGofStatistic],
)
def test_infinite_count_penalties_have_valid_monte_carlo_ranks(cls):
    # More cells than observations guarantees empty cells in every replicate.
    obj = cls(E.DEFAULT.parse({}), binning=EqualProbabilityBins(count=5))
    result = monte_carlo_exponential(obj, [0.3, 1.5], n_resamples=19, random_state=7)
    assert result.statistic == np.inf
    assert result.critical_value == np.inf
    assert result.p_value == 1
    assert not result.rejected


def test_monte_carlo_nan_replicates_are_errors(monkeypatch):
    obj = exp.GreenwoodExponentialityGofStatistic(E.DEFAULT.parse({}))
    values = iter([0.5, np.nan])
    monkeypatch.setattr(obj, "execute_statistic", lambda x: next(values))
    with pytest.raises(ValueError, match="NaN simulated"):
        monte_carlo_exponential(obj, [1, 2], n_resamples=1)


def test_hollander_proschan_rejects_nbu_in_lower_tail():
    # Uniform(1,2) is NBU and X>Y+Z is impossible, whereas the null mean is 1/4.
    obj = exp.HollanderProshanExponentialityGofStatistic(E.DEFAULT.parse({}))
    sample = np.linspace(1.1, 1.9, 40)
    assert obj.execute_statistic(sample) == 0
    assert obj.alternative().type().name == "LEFT"
    result = monte_carlo_exponential(obj, sample, n_resamples=99, random_state=7)
    assert result.p_value == pytest.approx(0.01)
    assert result.critical_value > 0
    assert result.rejected


@pytest.mark.parametrize("name", ["Gnedenko", "Harris"])
@pytest.mark.parametrize("ratio", [1e-6, 1.0, 1e6])
def test_spacing_f_ratios_use_both_tails(name, ratio):
    n, r = 20, 3
    numerator = np.zeros(n, dtype=bool)
    numerator[:r] = True
    if name == "Harris":
        numerator[-r:] = True
    spacings = np.where(numerator, ratio, 1.0)
    sample = np.cumsum(spacings / np.arange(n, 0, -1))
    obj = getattr(exp, name + "ExponentialityGofStatistic")(E.DEFAULT.parse({}), r=r)
    assert obj.execute_statistic(sample) == pytest.approx(ratio)
    assert obj.alternative().type().name == "TWO_TAILED"
    df1, df2 = 2 * numerator.sum(), 2 * (~numerator).sum()
    exact_p = min(1.0, 2 * min(stats.f.cdf(ratio, df1, df2), stats.f.sf(ratio, df1, df2)))
    result = monte_carlo_exponential(obj, sample, n_resamples=199, random_state=7)
    assert result.rejected == (exact_p <= 0.05)
    lower, upper = result.critical_value
    if ratio < 1:
        assert result.statistic < lower
        assert result.p_value == pytest.approx(0.01)
    elif ratio > 1:
        assert result.statistic > upper
        assert result.p_value == pytest.approx(0.01)
    else:
        assert lower < result.statistic < upper


@pytest.mark.parametrize("sample, expected", [([1, 3], 1 / 8), ([1, 2, 3], 1 / 18), ([2, 2, 2], 0)])
def test_we_published_normalization_and_greenwood_identity(sample, expected):
    parameters = E.DEFAULT.parse({})
    we = exp.WeExponentialityGofStatistic(parameters)
    greenwood = exp.GreenwoodExponentialityGofStatistic(parameters)
    for scale in [1e-150, 1.0, 1e150]:
        x = np.asarray(sample) * scale
        actual = we.execute_statistic(x)
        assert actual == pytest.approx(expected)
        assert actual == pytest.approx(greenwood.execute_statistic(x) - 1 / len(x), abs=1e-15)
