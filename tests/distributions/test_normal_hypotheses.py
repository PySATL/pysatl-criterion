import inspect

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.persistence.models.distribution_key import DistributionKey
from pysatl_criterion.persistence.stores.base import IStoreReader
from pysatl_criterion.statistics.goodness_of_fit import normal
from pysatl_criterion.utils.generator import get_hypothesis_generator


NORMAL_STATISTICS = [
    cls
    for _, cls in inspect.getmembers(normal, inspect.isclass)
    if cls.__module__ == normal.__name__
    and issubclass(cls, normal.AbstractNormalityGofStatistic)
    and not inspect.isabstract(cls)
]
SPECIFIED_STATISTICS = [
    normal.KolmogorovSmirnovNormalityGofStatistic,
    normal.CramerVonMiseNormalityGofStatistic,
]
GRAPH_STATISTICS = [
    cls for cls in NORMAL_STATISTICS if issubclass(cls, normal.AbstractGraphNormalityGofStatistic)
]
FAMILY_STATISTICS = [
    cls for cls in NORMAL_STATISTICS if cls not in SPECIFIED_STATISTICS + GRAPH_STATISTICS
]


@pytest.mark.parametrize("statistic_class", NORMAL_STATISTICS)
def test_normal_statistics_declare_only_fixed_hypothesis_parameters(statistic_class):
    statistic = statistic_class()
    if statistic_class in SPECIFIED_STATISTICS:
        expected = {"mean": 0, "var": 1}
    elif statistic_class in GRAPH_STATISTICS:
        expected = {"var": 1}
    else:
        expected = {}
    assert statistic.hypothesis().parameters() == expected


@pytest.mark.parametrize("statistic_class", FAMILY_STATISTICS)
def test_normal_family_rejects_fixed_distribution_parameters(statistic_class):
    for parameters in ({"mean": 3}, {"var": 4}):
        with pytest.raises(TypeError):
            statistic_class(**parameters)


@pytest.mark.parametrize("statistic_class", FAMILY_STATISTICS)
def test_normal_family_statistics_are_invariant_to_location_and_scale(statistic_class):
    sample = np.random.default_rng(101).normal(size=32)
    statistic = statistic_class()
    expected = statistic.execute_statistic(sample.copy())
    actual = statistic.execute_statistic(5 + 2 * sample)
    assert np.isfinite(expected)
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("statistic_class", GRAPH_STATISTICS)
def test_normal_graphs_use_known_variance_and_unknown_mean(statistic_class):
    statistic = statistic_class(var=4)
    assert statistic.hypothesis().parameters() == {"var": 4}
    sampler = get_hypothesis_generator(statistic)
    assert sampler.parameters() == {"mean": 0, "var": 4}
    actual = sampler.generate(32, random_state=np.random.default_rng(42))
    expected = stats.norm.rvs(loc=0, scale=2, size=32, random_state=np.random.default_rng(42))
    np.testing.assert_allclose(actual, expected)

    sample = np.random.default_rng(101).normal(size=32)
    assert statistic.execute_statistic(sample.copy()) == statistic.execute_statistic(sample + 5)
    with pytest.raises(TypeError):
        statistic_class(mean=3)
    with pytest.raises(TypeError):
        statistic_class(4)


@pytest.mark.parametrize("statistic_class", GRAPH_STATISTICS)
@pytest.mark.parametrize("var", [0, -1, np.nan, np.inf, -np.inf])
def test_normal_graphs_reject_invalid_variance(statistic_class, var):
    with pytest.raises(ValueError, match="var must be positive and finite"):
        statistic_class(var=var)


@pytest.mark.parametrize(
    "statistic",
    [
        normal.ShapiroWilkNormalityGofStatistic(),
        normal.RyanJoinerNormalityGofStatistic(),
        *(cls(var=4) for cls in GRAPH_STATISTICS),
    ],
)
def test_normal_family_and_graph_monte_carlo_accept_numpy_samples(statistic):
    result = MonteCarloLimitDistributionResolver(3).resolve(statistic, sample_size=16)
    assert len(result) == 3
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "statistic, parameters",
    [
        (normal.ShapiroWilkNormalityGofStatistic(), {}),
        (normal.RyanJoinerNormalityGofStatistic(), {}),
        (normal.GraphEdgesNumberNormalityGofStatistic(var=4), {"var": 4}),
        (normal.KolmogorovSmirnovNormalityGofStatistic(mean=3, var=4), {"mean": 3, "var": 4}),
        (normal.CramerVonMiseNormalityGofStatistic(mean=3, var=4), {"mean": 3, "var": 4}),
    ],
)
def test_normal_storage_lookup_uses_the_hypothesis_parameters(mocker, statistic, parameters):
    storage = mocker.Mock(spec=IStoreReader)
    storage.get.return_value = None
    assert StorageLimitDistributionResolver(storage).resolve(statistic, sample_size=32) is None
    storage.get.assert_called_once_with(DistributionKey(statistic.code(), parameters, 32))


def test_normal_family_sampler_uses_a_representative_without_fixing_the_hypothesis():
    statistic = normal.ShapiroWilkNormalityGofStatistic()
    sampler = get_hypothesis_generator(statistic)
    assert sampler.parameters() == {"mean": 0, "var": 1}
    assert statistic.hypothesis().parameters() == {}


def test_graph_independence_accepts_arrays_without_reordering_the_input():
    sample = np.array([0.5, -1.0, 1.5, -0.25, 0.75])
    original = sample.copy()
    statistic = normal.GraphIndependenceNumberNormalityGofStatistic(var=2)
    assert statistic.execute_statistic(sample) == statistic.execute_statistic(sample.tolist())
    np.testing.assert_array_equal(sample, original)
    with pytest.raises(ValueError):
        statistic.execute_statistic(np.array([]))
