"""Independent formula and contract regressions for every Gamma statistic."""

import doctest
import inspect
from itertools import combinations, pairwise

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous import gamma as g
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import gamma_statistics
from tests.parameter_cases import parameters_for


CLASSES = [
    cls
    for name, cls in vars(g).items()
    if name.endswith("GammaGofStatistic") and not name.startswith("Abstract")
]
SAMPLE = np.array([0.2, 0.7, 1.3, 2.1, 3.4])


@pytest.mark.parametrize("cls", CLASSES)
def test_common_interface_and_independent_calls(cls):
    statistic = cls(parameters_for(cls))
    x = SAMPLE.copy()
    state = vars(statistic).copy()
    first = statistic.execute_statistic(x, unused=True)
    statistic.execute_statistic(x * 2)
    assert isinstance(first, (float, np.float64))
    assert statistic.execute_statistic(x) == first
    np.testing.assert_array_equal(x, SAMPLE)
    assert vars(statistic) == state
    assert "kwargs" in inspect.signature(statistic.execute_statistic).parameters
    assert statistic.hypothesis().parameters() == ({"alfa": 1.0, "beta": 1.0})


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "sample",
    [
        [],
        [[1.0, 2.0]],
        1.0,
        [np.nan],
        [np.inf],
        [-1.0],
        [1j],
        np.ma.array([1.0, 2.0], mask=[False, True]),
    ],
)
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls(parameters_for(cls)).execute_statistic(sample)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, [1.0], 1j, True, "1"])
@pytest.mark.parametrize("parameter", ["alfa", "beta"])
def test_invalid_parameters(cls, parameter, value):
    with pytest.raises(ValueError):
        cls(parameters_for(cls, **{parameter: value}))


@pytest.mark.parametrize(
    "direction, scipy_direction",
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_reference_and_tail(direction, scipy_direction):
    statistic = g.KolmogorovSmirnovGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 3}),
        alternative_type=direction,
    )
    expected = stats.ks_1samp(
        SAMPLE, stats.gamma(2.5, scale=1 / 3).cdf, alternative=scipy_direction
    ).statistic
    assert statistic.execute_statistic(SAMPLE) == pytest.approx(expected)
    assert isinstance(statistic.alternative(), RightAlternative)


def test_edf_independent_uniform_formulas():
    u = np.array([0.03, 0.2, 0.55, 0.8, 0.95])
    x = stats.gamma.ppf(u, a=2.5, scale=1 / 3)
    n, i = len(u), np.arange(1, len(u) + 1)
    w = 1 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
    values = {
        g.AndersonDarlingGammaGofStatistic: -n
        - np.sum((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))) / n,
        g.CramerVonMisesGammaGofStatistic: w,
        g.WatsonGammaGofStatistic: w - n * (u.mean() - 0.5) ** 2,
        g.KuiperGammaGofStatistic: max(i / n - u) + max(u - (i - 1) / n),
        g.GreenwoodGammaGofStatistic: np.sum(np.diff(np.r_[0, u, 1]) ** 2),
        g.MoranGammaGofStatistic: -np.log(n * np.diff(np.r_[0, u, 1])).sum(),
        g.MinToshiyukiGammaGofStatistic: np.sum(
            np.maximum(i / n - u, u - (i - 1) / n) / np.sqrt(u * (1 - u))
        )
        / np.sqrt(n),
    }
    for cls, expected in values.items():
        assert cls(parameters_for(cls, alfa=2.5, beta=3)).execute_statistic(x) == pytest.approx(
            expected
        )


def test_exponential_tail_log_spacings():
    # Gamma(shape=1) has survival exp(-x): exact independent tail formula.
    x = np.array([40.0, 41.0, 42.0])
    logs = np.array(
        [np.log1p(-np.exp(-40)), -40 + np.log1p(-np.exp(-1)), -41 + np.log1p(-np.exp(-1)), -42]
    )
    expected = -logs.sum() - 4 * np.log(3)
    assert g.MoranGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1})
    ).execute_statistic(x) == pytest.approx(expected)
    assert np.isfinite(
        g.MinToshiyukiGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1})
        ).execute_statistic(x)
    )
    for cls in (
        g.AndersonDarlingGammaGofStatistic,
        g.MoranGammaGofStatistic,
        g.MinToshiyukiGammaGofStatistic,
    ):
        assert cls(parameters_for(cls)).execute_statistic([0, 1]) == np.inf
    assert (
        g.MoranGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1})
        ).execute_statistic([2, 2])
        == np.inf
    )


@pytest.mark.parametrize("bins", [True, 1, 2.5, np.nan, np.inf, [3]])
def test_bin_validation(bins):
    with pytest.raises((ValueError, TypeError)):
        g.Chi2PearsonGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1}),
            binning=EqualProbabilityBins(count=bins),
        )


@pytest.mark.parametrize(
    "cls,power",
    [
        (g.NeymanChi2GammaGofStatistic, -2),
        (g.ModifiedLogLikelihoodGammaGofStatistic, -1),
        (g.FreemanTukeyGammaGofStatistic, -0.5),
        (g.LikelihoodRatioGammaGofStatistic, 0),
        (g.CressieReadGammaGofStatistic, 2 / 3),
        (g.Chi2PearsonGammaGofStatistic, 1),
    ],
)
def test_histogram_divergence_zero_bins(cls, power):
    # Four quantile cells contain 3, 0, 1, 2 points, independently prescribed.
    x = stats.gamma.ppf([0.1, 0.12, 0.2, 0.6, 0.8, 0.9], a=2.5, scale=0.5)
    o, e = np.array([3.0, 0, 1, 2]), np.full(4, 1.5)
    if power <= -1:
        expected = np.inf
    elif power == 0:
        mask = o > 0
        expected = 2 * np.sum(o[mask] * np.log(o[mask] / e[mask]))
    else:
        # Algebraically distinct continuous-limit expression, including O=0.
        expected = 2 * (np.sum(o ** (power + 1) / e**power) - np.sum(o)) / (power * (power + 1))
    options = {"bins": 4} if power == 0 else {"binning": EqualProbabilityBins(count=4)}
    value = cls(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 2}), **options
    ).execute_statistic(x)
    assert value == pytest.approx(expected)


def test_calibration_storage_settings(mocker):
    store = mocker.Mock()
    resolver = StorageLimitDistributionResolver(store)
    for stat in (
        g.LikelihoodRatioGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1})
        ),
        g.KolmogorovSmirnovGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1}),
            alternative_type=AlternativeType.LEFT,
        ),
    ):
        with pytest.raises(ValueError):
            resolver.resolve(stat, 10)
    store.get.assert_not_called()
    store.get.return_value = None
    stat = g.Chi2PearsonGammaGofStatistic(parameters_for(g.Chi2PearsonGammaGofStatistic))
    assert resolver.resolve(stat, 10) is None
    assert "binning" in stat.configuration()
    store.get.assert_called_once()
    values = MonteCarloLimitDistributionResolver(3).resolve(
        g.Chi2PearsonGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1})
        ),
        10,
    )
    assert len(values) == 3


def test_ppcc_reference_and_scale_extremes():
    q = stats.gamma.ppf((np.arange(1, 6) - 0.375) / 5.25, a=2.5)
    expected = 1 - stats.pearsonr(SAMPLE, q).statistic
    statistic = g.ProbabilityPlotCorrelationGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1e-300})
    )
    for scale in (1.0, 1e300, 1e-300):
        assert statistic.execute_statistic(SAMPLE * scale) == pytest.approx(expected, abs=1e-14)
    assert isinstance(statistic.alternative(), RightAlternative)


@pytest.mark.parametrize(
    "u", [[0.2], [0.4, 0.4, 0.4], [0.1, 0.8, 0.81, 0.82], [0.1, 0.15, 0.2, 0.5, 0.51, 0.52, 0.9]]
)
def test_graphs_against_enumeration(u):
    x = stats.gamma.ppf(u, a=2)
    # Use actual transformed values to preserve the strict floating-point boundary.
    transformed = stats.gamma.cdf(x, a=2)
    h = np.ptp(transformed) / 10
    graph = nx.Graph()
    graph.add_nodes_from(range(len(u)))
    graph.add_edges_from(
        (i, j)
        for i, j in combinations(range(len(u)), 2)
        if abs(transformed[i] - transformed[j]) < h
    )
    expected = {
        g.GraphEdgesNumberGammaGofStatistic: graph.number_of_edges(),
        g.GraphMaxDegreeGammaGofStatistic: max(dict(graph.degree()).values()),
        g.GraphAverageDegreeGammaGofStatistic: 2 * graph.number_of_edges() / len(u),
        g.GraphConnectedComponentsGammaGofStatistic: nx.number_connected_components(graph),
        g.GraphCliqueNumberGammaGofStatistic: max(map(len, nx.find_cliques(graph))),
        g.GraphIndependenceNumberGammaGofStatistic: max(
            map(len, nx.find_cliques(nx.complement(graph)))
        ),
    }
    for cls, value in expected.items():
        assert cls(parameters_for(cls, alfa=2)).execute_statistic(x) == float(value)


def test_doc_examples():
    assert doctest.testmod(g, extraglobs=vars(gamma_statistics)).failed == 0


@pytest.mark.parametrize("shape,rate", [(0.4, 3), (1, 0.5), (2.5, 2), (20, 0.1)])
def test_generated_edf_independent_definitions(shape, rate):
    # Prescribed probabilities give references independent of the shared kernels.
    u = np.array([0.002, 0.08, 0.27, 0.44, 0.81, 0.98])
    x = stats.gamma.ppf(u, a=shape, scale=1 / rate)
    n = len(u)
    i = np.arange(1, n + 1)
    r = i - 0.5
    dp, dm = max(i / n - u), max(u - (i - 1) / n)
    local = np.maximum(i / n - u, u - (i - 1) / n)
    references = {
        "KolmogorovSmirnovDPlus": dp,
        "KolmogorovSmirnovDMinus": dm,
        "KolmogorovSmirnov": max(dp, dm),
        "AndersonDarling": -n - sum((2 * i - 1) * np.log(u * (1 - u[::-1]))) / n,
        "CramerVonMises": 1 / (12 * n) + sum((u - r / n) ** 2),
        "Watson": 1 / (12 * n) + sum((u - r / n) ** 2) - n * (u.mean() - 0.5) ** 2,
        "Kuiper": dp + dm,
        "KuiperScaled": np.sqrt(n) * (dp + dm),
        "KuiperCorrected": (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * (dp + dm),
        "LiaoShimokawa": sum(local / np.sqrt(u * (1 - u))) / np.sqrt(n),
        "ZhangK": max(r * np.log(r / (n * u)) + (n - r) * np.log((n - r) / (n * (1 - u)))),
        "ZhangA": -sum(np.log(u) / (n - r) + np.log(1 - u) / r),
        "ZhangC": sum(np.log(((1 - u) / u) / ((n - i + 0.25) / (i - 0.75))) ** 2),
    }
    assert {name + "GammaGofStatistic" for name in references} == set(
        gamma_statistics._EDF_STATISTICS
    )
    for name, expected in references.items():
        cls = getattr(g, name + "GammaGofStatistic")
        assert cls(parameters_for(cls, alfa=shape, beta=rate)).execute_statistic(
            x[::-1]
        ) == pytest.approx(expected)
    assert g.MinToshiyukiGammaGofStatistic(
        parameters_for(g.MinToshiyukiGammaGofStatistic, alfa=shape, beta=rate)
    ).execute_statistic(x) == pytest.approx(references["LiaoShimokawa"])


@pytest.mark.parametrize("cls", CLASSES)
def test_change_of_rate_units(cls):
    original = cls(parameters_for(cls, alfa=2.5, beta=3)).execute_statistic(SAMPLE)
    rescaled = cls(parameters_for(cls, alfa=2.5, beta=3 / 7)).execute_statistic(SAMPLE * 7)
    assert rescaled == pytest.approx(original, abs=1e-13)


@pytest.mark.parametrize(
    "name,power",
    [
        ("Chi2Pearson", 1),
        ("FreemanTukey", -0.5),
        ("ModifiedLogLikelihood", -1),
        ("NeymanChi2", -2),
        ("CressieRead", 2 / 3),
    ],
)
def test_frequency_nonempty_counts_scipy(name, power):
    probabilities = [0.1, 0.2, 0.3, 0.55, 0.6, 0.9]
    x = stats.gamma.ppf(probabilities, a=2.5, scale=0.5)
    cls = getattr(g, name + "GammaGofStatistic")
    value = cls(
        parameters_for(cls, alfa=2.5, beta=2), binning=EqualProbabilityBins(count=3)
    ).execute_statistic(x)
    assert value == pytest.approx(
        stats.power_divergence([3, 2, 1], [2, 2, 2], lambda_=power).statistic
    )


def test_moran_normalization_and_ppcc_blind_spot():
    u = np.arange(1, 6) / 6
    x = stats.gamma.ppf(u, a=2)
    cls = g.MoranGammaGofStatistic
    assert cls(parameters_for(cls, alfa=2)).execute_statistic(x) == pytest.approx(6 * np.log(6 / 5))
    cls = g.ProbabilityPlotCorrelationGammaGofStatistic
    statistic = cls(parameters_for(cls, alfa=2))
    assert statistic.execute_statistic(SAMPLE) == pytest.approx(
        statistic.execute_statistic(3 + 7 * SAMPLE)
    )
    assert statistic.execute_statistic([1, 2]) == pytest.approx(0, abs=1e-15)


def test_gamma_documentation_coverage_and_examples():
    import re
    from pathlib import Path

    directory = Path(__file__).resolve().parents[2] / "docs/criteria/goodness-of-fit/gamma"
    index = (directory / "index.md").read_text()
    navigation = (directory.parents[3] / "mkdocs.yml").read_text()
    pages = list(directory.glob("*_criterion.md"))
    assert len(pages) == len(CLASSES) == 29
    for cls in CLASSES:
        matches = [page for page in pages if cls.__name__ in page.read_text()]
        assert len(matches) == 1, cls.__name__
        page = matches[0]
        body = page.read_text()
        assert page.name in index and page.name in navigation
        assert "## Source" in body and "https://" in body
        for code in re.findall(r"```python\n(.*?)```", body, re.DOTALL):
            namespace = {}
            exec(compile(code, str(page), "exec"), namespace)  # noqa: S102
            assert np.isfinite(namespace["value"])


def test_edf_integrals_independent_quadrature():
    from scipy.integrate import quad

    u = np.array([0.01, 0.13, 0.42, 0.75, 0.98])
    edges = np.r_[0, u, 1]
    n = len(u)
    squared = weighted = signed = 0.0
    for j, (left, right) in enumerate(pairwise(edges)):
        squared += quad(lambda t, j=j: (j / n - t) ** 2, left, right)[0]
        weighted += quad(lambda t, j=j: (j / n - t) ** 2 / (t * (1 - t)), left, right)[0]
        signed += quad(lambda t, j=j: j / n - t, left, right)[0]
    x = stats.gamma.ppf(u, a=2.5, scale=0.5)
    for cls, expected in (
        (g.CramerVonMisesGammaGofStatistic, n * squared),
        (g.AndersonDarlingGammaGofStatistic, n * weighted),
        (g.WatsonGammaGofStatistic, n * (squared - signed**2)),
    ):
        assert cls(parameters_for(cls, alfa=2.5, beta=2)).execute_statistic(x) == pytest.approx(
            expected
        )


def test_logarithmic_edf_exponential_tail():
    x = np.array([40.0, 41.0, 42.0])
    n = len(x)
    i = np.arange(1, n + 1)
    r = i - 0.5
    # Gamma(shape=1, rate=1): log survival is exactly -x.
    log_u = np.log1p(-np.exp(-x))
    references = {
        g.ZhangKGammaGofStatistic: max(
            r * (np.log(r / n) - log_u) + (n - r) * (np.log((n - r) / n) + x)
        ),
        g.ZhangAGammaGofStatistic: -sum(log_u / (n - r) - x / r),
        g.ZhangCGammaGofStatistic: sum((-x - log_u - np.log((n - i + 0.25) / (i - 0.75))) ** 2),
        g.AndersonDarlingGammaGofStatistic: -n - sum((2 * i - 1) * (log_u - x[::-1])) / n,
    }
    for cls, expected in references.items():
        assert cls(parameters_for(cls)).execute_statistic(x) == pytest.approx(expected)
    lhs = g.LiaoShimokawaGammaGofStatistic(parameters_for(g.LiaoShimokawaGammaGofStatistic))
    rhs = g.MinToshiyukiGammaGofStatistic(parameters_for(g.MinToshiyukiGammaGofStatistic))
    assert lhs.execute_statistic(x) == pytest.approx(rhs.execute_statistic(x))


@pytest.mark.parametrize("seed", range(10))
def test_random_graphs_by_exhaustive_enumeration(seed):
    u = np.sort(np.random.default_rng(seed).uniform(size=8))
    test_graphs_against_enumeration(u)
