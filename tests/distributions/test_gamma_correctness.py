"""Independent formula and contract regressions for every Gamma statistic."""

import doctest
import inspect
from itertools import combinations

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution.distributions import GammaDistributionDescriptor
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType, RightAlternative
from pysatl_criterion.statistics.goodness_of_fit import gamma as g
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
    with pytest.raises(ValueError):
        g.Chi2PearsonGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1}), bins=bins
        )


@pytest.mark.parametrize("power", [-2.0, -1.0, -0.5, 0.0, 2 / 3, 1.0, 2.0])
def test_histogram_divergence_zero_bins(power):
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
    value = g.CressieReadGammaGofStatistic(
        GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 2}), power=power, bins=4
    ).execute_statistic(x)
    assert value == pytest.approx(expected)


def test_calibration_storage_settings(mocker):
    store = mocker.Mock()
    resolver = StorageLimitDistributionResolver(store)
    for stat in (
        g.Chi2PearsonGammaGofStatistic(
            GammaDistributionDescriptor.DEFAULT.parse({"alfa": 1, "beta": 1})
        ),
        g.CressieReadGammaGofStatistic(
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
    assert doctest.testmod(g).failed == 0
