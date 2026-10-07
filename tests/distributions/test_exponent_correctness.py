"""Independent formula, contract and boundary checks for exponential statistics."""

import doctest
import inspect
import itertools
import math
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest
from scipy import integrate, special, stats

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import exponent as exp
from pysatl_criterion.utils.generator import get_hypothesis_generator


CLASSES = [
    cls
    for name, cls in vars(exp).items()
    if inspect.isclass(cls) and cls.__module__ == exp.__name__ and not name.startswith("Abstract")
]
FIXED = (
    exp.KolmogorovSmirnovExponentialityGofStatistic,
    exp.CramerVonMisesExponentialityGofStatistic,
)
GRAPHS = [cls for cls in CLASSES if cls.__name__.startswith("Graph")]
COMPOSITE = [cls for cls in CLASSES if cls not in FIXED]


@pytest.mark.parametrize("cls", CLASSES)
def test_interface_nonmutation_and_statelessness(cls):
    sample = np.array([2.0, 0.25, 0.5, 1.0])
    before = sample.copy()
    obj = cls()
    state = vars(obj).copy()
    value = obj.execute_statistic(sample)
    obj.execute_statistic([0.0, 0.125, 0.75, 4.0])
    assert isinstance(value, float)
    assert not math.isnan(value)
    np.testing.assert_array_equal(sample, before)
    assert vars(obj) == state
    assert obj.execute_statistic(tuple(before)) == value
    sample.setflags(write=False)
    assert obj.execute_statistic(sample) == value
    with pytest.raises(TypeError):
        obj.execute_statistic(sample, unknown_option=True)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "sample", [[], [[1, 2]], [1, np.nan], [1, np.inf], [-1, 2, 3, 4], [1 + 1j, 2]]
)
def test_invalid_samples(cls, sample):
    with pytest.raises(ValueError):
        cls().execute_statistic(sample)


@pytest.mark.parametrize("cls", COMPOSITE)
def test_scale_invariance_and_unknown_rate(cls):
    sample = np.array([0.25, 0.5, 1, 2, 4.0])
    obj = cls()
    expected = obj.execute_statistic(sample)
    for scale in [2.0**-1000, 2.0**1000, 1.0 / 8, 8]:
        assert obj.execute_statistic(sample * scale) == pytest.approx(expected, abs=1e-14)
    assert obj.hypothesis().parameters() == {}
    with pytest.raises(TypeError):
        cls(lam=2)


@pytest.mark.parametrize("cls", FIXED)
@pytest.mark.parametrize("lam", [0, -1, np.inf, np.nan, [1], 1j, True, None])
def test_invalid_fixed_rate(cls, lam):
    with pytest.raises(ValueError):
        cls(lam=lam)


@pytest.mark.parametrize(
    "direction,scipy_direction",
    [
        (AlternativeType.TWO_TAILED, "two-sided"),
        (AlternativeType.RIGHT, "greater"),
        (AlternativeType.LEFT, "less"),
    ],
)
def test_ks_matches_known_cdf(direction, scipy_direction):
    x = np.array([0, 0.05, 0.1, 0.7, 2])
    obj = exp.KolmogorovSmirnovExponentialityGofStatistic(direction, lam=3)
    expected = stats.kstest(x, stats.expon(scale=1 / 3).cdf, alternative=scipy_direction).statistic
    assert obj.execute_statistic(x) == pytest.approx(expected)
    assert obj.hypothesis().parameters() == {"lam": 3}
    assert obj.alternative().type() == AlternativeType.RIGHT
    assert obj.execute_statistic(x) != exp.KolmogorovSmirnovExponentialityGofStatistic(
        direction, lam=1
    ).execute_statistic(x)


def test_cvm_by_integrating_empirical_cdf_on_probability_scale():
    x = np.array([0, 0.2, 0.7, 1.5])
    u = stats.expon(scale=0.5).cdf(x)
    cuts = np.r_[0, u, 1]
    integral = sum(
        integrate.quad(lambda t, k=k: (k / len(x) - t) ** 2, a, b)[0]
        for k, (a, b) in enumerate(itertools.pairwise(cuts))
    )
    obj = exp.CramerVonMisesExponentialityGofStatistic(lam=2)
    assert obj.execute_statistic(x) == pytest.approx(len(x) * integral)
    assert obj.hypothesis().parameters() == {"lam": 2}


@pytest.mark.parametrize(
    "cls,power",
    [
        (exp.HegazyGreen1ExponentialityGofStatistic, 1),
        (exp.HegazyGreen2ExponentialityGofStatistic, 2),
    ],
)
def test_fitted_quantile_distances(cls, power):
    for x in ([0.25, 0.5, 1, 4], [0.1, 0.3, 0.9, 1.1, 1.5]):
        q = stats.expon(scale=np.mean(x)).ppf(np.arange(1, len(x) + 1) / (len(x) + 1))
        expected = sum(abs(a - b) ** power for a, b in zip(sorted(x), q, strict=True))
        expected /= len(x) * np.mean(x) ** power
        assert cls().execute_statistic(x) == pytest.approx(expected)


@pytest.mark.parametrize("cls", GRAPHS)
@pytest.mark.parametrize(
    "sample",
    [[0], [0, 0, 0], [1, 1, 1], [0, 0.125, 0.25, 2], [0, 1, 2, 10], [1, 1.01, 1.02, 1.03, 2]],
)
def test_graphs_against_explicit_networkx_graph(cls, sample):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sample)))
    threshold = (max(sample) - min(sample)) / 10
    graph.add_edges_from(
        (i, j)
        for i, j in itertools.combinations(range(len(sample)), 2)
        if abs(sample[i] - sample[j]) < threshold
    )
    expected = {
        "EDGESNUMBER": graph.number_of_edges(),
        "MAXDEGREE": max(dict(graph.degree).values()),
        "AVGDEGREE": 2 * graph.number_of_edges() / len(sample),
        "CONNECTEDCOMPONENTS": nx.number_connected_components(graph),
        "CLIQUENUMBER": max(map(len, nx.find_cliques(graph))),
        "INDEPENDENCENUMBER": max(map(len, nx.find_cliques(nx.complement(graph)))),
    }
    assert cls().execute_statistic(sample) == expected[cls.short_code()]


def test_characterizations_by_integer_enumeration():
    x = list(range(1, 8))
    n = len(x)
    ahs = (
        sum((abs(a - b) < c) - (2 * min(a, b) < c) for a, b, c in itertools.product(x, repeat=3))
        / n**3
    )
    hp = sum(x[i] > x[j] + x[k] for i, j, k in itertools.permutations(range(n), 3))
    hp /= n * (n - 1) * (n - 2)
    h = sum(sorted(t)[1] - min(t) < v for t in itertools.combinations(x, 3) for v in x)
    g = sum(min(t) < v for t in itertools.combinations(x, 2) for v in x)
    rossberg = h / (n * math.comb(n, 3)) - g / (n * math.comb(n, 2))
    assert exp.AhsanullahExponentialityGofStatistic().execute_statistic(x) == ahs
    assert exp.HollanderProshanExponentialityGofStatistic().execute_statistic(x) == pytest.approx(
        hp
    )
    assert exp.RossbergExponentialityGofStatistic().execute_statistic(x) == pytest.approx(rossberg)


def test_spacing_ratios_from_prescribed_independent_spacings():
    d = np.array([2, 3, 5, 7, 11, 13.0])
    x = np.cumsum(d / np.arange(6, 0, -1))
    assert exp.GnedenkoExponentialityGofStatistic(r=2).execute_statistic(x) == pytest.approx(
        2.5 / 9
    )
    assert exp.HarrisExponentialityGofStatistic(r=2).execute_statistic(x) == pytest.approx(7.25 / 6)
    expected = 12 * (math.log(sum(d) / 6) - sum(map(math.log, d)) / 6) / (1 + 7 / 36)
    assert exp.EpsteinExponentialityGofStatistic().execute_statistic(x) == pytest.approx(expected)


@pytest.mark.parametrize(
    "cls", [exp.CoxOakesExponentialityGofStatistic, exp.MoranExponentialityGofStatistic]
)
def test_logarithmic_boundary(cls):
    with np.errstate(all="raise"):
        assert cls().execute_statistic([0, 1, 2]) == -math.inf


def test_ratio_and_spacing_boundaries():
    with np.errstate(all="raise"):
        assert exp.EpsteinExponentialityGofStatistic().execute_statistic([1, 1, 2]) == math.inf
        assert exp.WongWongExponentialityGofStatistic().execute_statistic([0, 1]) == math.inf
        assert exp.GnedenkoExponentialityGofStatistic(r=1).execute_statistic([1, 1]) == math.inf
        assert exp.HarrisExponentialityGofStatistic(r=1).execute_statistic([1, 1, 2]) == math.inf
        with pytest.raises(ValueError):
            exp.ShapiroWilkExponentialityGofStatistic().execute_statistic([1, 1, 1])


@pytest.mark.parametrize("cls", [cls for cls in COMPOSITE if cls not in GRAPHS])
def test_degenerate_sample_rejected(cls):
    with pytest.raises(ValueError):
        cls().execute_statistic([0, 0, 0, 0])
    with pytest.raises(ValueError):
        cls().execute_statistic([1])


@pytest.mark.parametrize(
    "cls", [exp.HarrisExponentialityGofStatistic, exp.GnedenkoExponentialityGofStatistic]
)
def test_split_validation(cls):
    for r in [0, -1, 1.5, True, np.inf, [1]]:
        with pytest.raises(ValueError):
            cls(r=r)
    with pytest.raises(ValueError):
        cls(r=4).execute_statistic([1, 2, 3, 4])


@pytest.mark.parametrize(
    "cls,key,invalid",
    [
        (exp.AtkinsonExponentialityGofStatistic, "p", [-1, -2, 0, 1, np.nan, [0.5], True]),
        (exp.LorenzExponentialityGofStatistic, "p", [0, 1, 2, np.inf, [0.5], True]),
        (exp.DeshpandeExponentialityGofStatistic, "b", [0, 1, 2, np.nan, [0.5], True]),
    ],
)
def test_scalar_settings(cls, key, invalid):
    for value in invalid:
        with pytest.raises(ValueError):
            cls(**{key: value})
    with pytest.raises(TypeError):
        cls().execute_statistic([1, 2, 3, 4], **{key: 0.5})


def test_moment_and_lorenz_settings():
    x = np.array([0.25, 0.5, 1, 2])
    for p in [-0.5, 0.5, 2]:
        expected = 2 * abs(np.mean(x**p) ** (1 / p) / np.mean(x) - special.gamma(1 + p) ** (1 / p))
        assert exp.AtkinsonExponentialityGofStatistic(p=p).execute_statistic(x) == pytest.approx(
            expected
        )
    assert exp.LorenzExponentialityGofStatistic(p=0.75).execute_statistic(x) == 1.75 / 3.75
    with pytest.raises(ValueError):
        exp.LorenzExponentialityGofStatistic(p=0.1).execute_statistic(x)


@pytest.mark.parametrize(
    "cls",
    [
        exp.HegazyGreen1ExponentialityGofStatistic,
        exp.HegazyGreen2ExponentialityGofStatistic,
        exp.KimberMichaelExponentialityGofStatistic,
        exp.EpsteinExponentialityGofStatistic,
    ],
)
def test_discrepancies_reject_large_values(cls):
    assert cls().alternative().type() == AlternativeType.RIGHT


def test_shapiro_wilk_is_two_tailed():
    assert (
        exp.ShapiroWilkExponentialityGofStatistic().alternative().type()
        == AlternativeType.TWO_TAILED
    )


def test_documented_examples():
    failures, count = doctest.testmod(exp)
    assert count == 3 * len(CLASSES)
    assert failures == 0


@pytest.mark.parametrize(
    "obj",
    [
        exp.AtkinsonExponentialityGofStatistic(p=0.5),
        exp.DeshpandeExponentialityGofStatistic(b=0.3),
        exp.LorenzExponentialityGofStatistic(p=0.75),
        exp.GnedenkoExponentialityGofStatistic(r=1),
        exp.HarrisExponentialityGofStatistic(r=1),
        exp.KolmogorovSmirnovExponentialityGofStatistic(AlternativeType.LEFT),
    ],
)
def test_storage_rejects_unidentified_settings(obj):
    store = Mock()
    with pytest.raises(ValueError, match="nondefault settings"):
        StorageLimitDistributionResolver(store).resolve(obj, 6)
    store.get.assert_not_called()


@pytest.mark.parametrize("cls", CLASSES)
def test_monte_carlo_calls_statistic_on_each_sample(cls, monkeypatch):
    obj = cls()
    x, y = [0.25, 0.5, 1, 4], [0.125, 0.25, 2, 3]
    sampler = Mock()
    sampler.generate.side_effect = [x, y]
    monkeypatch.setattr(
        "pysatl_criterion.hypothesis_testing.limit_distribution.base.get_hypothesis_generator",
        lambda statistic: sampler,
    )
    actual = MonteCarloLimitDistributionResolver(2).resolve(obj, 4)
    assert actual == [obj.execute_statistic(x), obj.execute_statistic(y)]
    assert sampler.generate.call_count == 2


@pytest.mark.parametrize(
    "obj",
    [
        exp.EppsPulleyExponentialityGofStatistic(),
        exp.KolmogorovSmirnovExponentialityGofStatistic(lam=3),
    ],
)
def test_null_sampler_is_supported(obj):
    sample = get_hypothesis_generator(obj).generate(5)
    assert len(sample) == 5
    assert np.all(np.asarray(sample) >= 0)


@pytest.mark.parametrize("p", [-1e-12, 1e-12, -1e-17, 1e-17])
def test_atkinson_near_zero_power(p):
    x = np.array([0.25, 0.5, 1, 2])
    limit = 2 * abs(stats.gmean(x) / np.mean(x) - math.exp(-np.euler_gamma))
    value = exp.AtkinsonExponentialityGofStatistic(p=p).execute_statistic(x)
    assert value == pytest.approx(limit, abs=1e-11)


def test_atkinson_negative_power_zero_uses_limit():
    p = -0.5
    reference = 2 * special.gamma(1 + p) ** (1 / p)
    assert exp.AtkinsonExponentialityGofStatistic(p=p).execute_statistic(
        [0, 1, 2, 3]
    ) == pytest.approx(reference)


def test_fixed_cdf_at_extreme_rate_and_boundary():
    for cls in FIXED:
        obj = cls(lam=1e308)
        with np.errstate(all="raise"):
            value = obj.execute_statistic([0, 1, 1e308])
        assert math.isfinite(value)
        assert math.isfinite(cls(lam=1e-308).execute_statistic([0, 0, 0]))
