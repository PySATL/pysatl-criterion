"""Generated frequency criteria, discovery, and calibration boundaries."""

import inspect
import pickle
from typing import get_origin
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import stats

from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    MonteCarloLimitDistributionResolver,
    StorageLimitDistributionResolver,
)
from pysatl_criterion.statistics.goodness_of_fit import frequency
from pysatl_criterion.statistics.goodness_of_fit.binning import (
    DiscreteCategoryBins,
    EqualProbabilityBins,
    ExplicitBins,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous import uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous._frequency_factory import (
    export_frequency_statistics,
)
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform.uniform_statistics import (
    _FREQUENCY_STATISTICS,
)
from pysatl_criterion.utils.statistic import get_available_criteria, get_criterion_by_code


@pytest.mark.parametrize("name", _FREQUENCY_STATISTICS)
def test_generated_classes(name):
    cls = getattr(uniform, name)
    assert not inspect.isabstract(cls)
    assert "short_code" not in cls.__dict__
    assert cls.__module__ == uniform.__name__
    assert cls.__qualname__ == name
    assert name in uniform.__all__
    assert pickle.loads(pickle.dumps(cls)) is cls  # noqa: S301 - locally generated class
    assert cls in get_available_criteria(Uniform.type())
    assert get_criterion_by_code(cls.code()) is cls
    parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
    sample = [0.1, 0.15, 0.4, 0.6, 0.8, 0.9]
    equal = cls(parameters, binning=EqualProbabilityBins(4))
    explicit = cls(parameters, binning=ExplicitBins((0, 0.25, 0.5, 0.75, 1)))
    assert equal.execute_statistic(sample) == pytest.approx(explicit.execute_statistic(sample))
    powers = {
        "CHI2_PEARSON": 1,
        "FREEMAN_TUKEY": -0.5,
        "MODIFIED_LOG_LIKELIHOOD": -1,
        "CHI2_NEYMAN": -2,
        "CRESSIE_READ": 2 / 3,
    }
    expected = stats.power_divergence([2, 1, 1, 2], [1.5] * 4, lambda_=powers[cls.short_code()])
    assert equal.execute_statistic(sample) == pytest.approx(expected.statistic)


def test_all_kernels_are_registered():
    kernels = {
        cls
        for _, cls in inspect.getmembers(frequency, inspect.isclass)
        if get_origin(cls) is None
        and issubclass(cls, frequency.AbstractFrequencyStatistic)
        and cls is not frequency.AbstractFrequencyStatistic
    }
    assert {s.algorithm for s in _FREQUENCY_STATISTICS.values()} == kernels


def test_storage_uses_binning_identity():
    from pysatl_criterion.persistence.models.distribution_key import DistributionKey

    store = Mock()
    store.get.return_value = None
    criterion = uniform.Chi2PearsonUniformGofStatistic(Uniform.DEFAULT.parse({"a": 0, "b": 1}))
    assert StorageLimitDistributionResolver(store).resolve(criterion, 20) is None
    store.get.assert_called_once_with(DistributionKey.from_statistic(criterion, 20))
    assert store.get.call_args.args[0].criterion_configuration["binning"]["type"] == (
        "equal_probability"
    )


def test_monte_carlo_rebuilds_binning_for_each_simulation():
    class CountingBins:
        def __init__(self):
            self.sizes = []

        def supports(self, kind):
            return True

        def build(self, sample, distribution):
            self.sizes.append(len(sample))
            return EqualProbabilityBins(3).build(sample, distribution)

    binning = CountingBins()
    criterion = uniform.Chi2PearsonUniformGofStatistic(
        Uniform.DEFAULT.parse({"a": 0, "b": 1}), binning=binning
    )
    values = MonteCarloLimitDistributionResolver(4).resolve(criterion, 20)
    assert len(values) == 4
    assert np.all(np.isfinite(values))
    assert binning.sizes == [20] * 4


def test_configuration_validation():
    parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
    cls = uniform.Chi2PearsonUniformGofStatistic
    with pytest.raises(TypeError):
        cls(parameters, bins=4)
    with pytest.raises(TypeError):
        cls(parameters, lambda_=1)
    with pytest.raises(ValueError, match="distribution kind"):
        cls(parameters, binning=DiscreteCategoryBins((1, 2)))
    with pytest.raises(ValueError):
        cls(Uniform.DEFAULT.parse({}))


def test_registration_is_atomic_on_name_collision():
    namespace = {"__name__": "example", "CressieReadUniformGofStatistic": object()}
    before = namespace.copy()
    with pytest.raises(ValueError):
        export_frequency_statistics(
            namespace, uniform.AbstractUniformGofStatistic, _FREQUENCY_STATISTICS
        )
    assert namespace == before


def test_configuration_roundtrip_through_real_storage():
    from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
    from pysatl_criterion.persistence.sqlalchemy.database import AlchemyDatabase
    from pysatl_criterion.persistence.sqlalchemy.stores.limit_distribution import (
        AlchemyLimitDistributionStorage,
    )

    parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
    cls = uniform.Chi2PearsonUniformGofStatistic
    criterion = cls(parameters, binning=EqualProbabilityBins(4))
    values = MonteCarloLimitDistributionResolver(5).resolve(criterion, 20)
    model = LimitDistributionModel.from_statistic(criterion, 20, values)
    db = AlchemyDatabase("sqlite:///:memory:")
    db.init()
    with db.session_factory() as session:
        store = AlchemyLimitDistributionStorage(session)
        store.insert(model)
        resolver = StorageLimitDistributionResolver(store)
        assert resolver.resolve(criterion, 20) == pytest.approx(values)
        assert resolver.resolve(cls(parameters, binning=EqualProbabilityBins(8)), 20) is None
    db.dispose()


def test_builtin_configurations_and_custom_guard():
    from pysatl_criterion.statistics.configuration import serialize_configuration
    from pysatl_criterion.statistics.goodness_of_fit.binning import (
        AlternativeOptimizedBins,
        EqualWidthBins,
        FixedCount,
        InterpolatedBins,
        PowerCount,
        SqrtCount,
        SturgesCount,
    )

    assert EqualProbabilityBins(4).to_config() == EqualProbabilityBins(FixedCount(4)).to_config()
    policies = [
        EqualProbabilityBins(PowerCount()),
        EqualProbabilityBins(SturgesCount()),
        EqualProbabilityBins(SqrtCount()),
        EqualWidthBins(0, 1, 4),
        InterpolatedBins(0, 1, 4),
        ExplicitBins((-np.inf, 0, np.inf)),
        DiscreteCategoryBins((0, 1)),
    ]
    for policy in policies:
        assert '"version":1' in serialize_configuration(policy.to_config())
    selected = AlternativeOptimizedBins(EqualProbabilityBins(4), 20, (0.4,))
    assert selected.to_config()["selected"] == EqualProbabilityBins(4).to_config()

    class CustomBins:
        def build(self, sample, distribution):
            return EqualProbabilityBins(4).build(sample, distribution)

        def supports(self, kind):
            return True

    criterion = uniform.Chi2PearsonUniformGofStatistic(
        Uniform.DEFAULT.parse({"a": 0, "b": 1}), binning=CustomBins()
    )
    with pytest.raises(TypeError, match="to_config"):
        criterion.configuration()


def test_custom_count_configuration_and_selected_provenance():
    from pysatl_criterion.statistics.goodness_of_fit.binning import AlternativeOptimizedBins

    class CustomCount:
        def resolve(self, sample_size):
            return 4

        def to_config(self):
            return {"type": "test_custom_count", "version": 1, "count": 4}

    policy = EqualProbabilityBins(CustomCount())
    assert policy.to_config()["count"] == CustomCount().to_config()
    assert AlternativeOptimizedBins(policy, 20, (0.1,)).to_config() == (
        AlternativeOptimizedBins(policy, 20, (0.9,)).to_config()
    )
    assert AlternativeOptimizedBins(policy, 20, ()).to_config() != (
        AlternativeOptimizedBins(policy, 30, ()).to_config()
    )


def test_custom_binning_without_config_cannot_be_saved():
    from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel

    class CustomBins:
        def supports(self, kind):
            return True

        def build(self, sample, distribution):
            return EqualProbabilityBins(4).build(sample, distribution)

    criterion = uniform.Chi2PearsonUniformGofStatistic(
        Uniform.DEFAULT.parse({"a": 0, "b": 1}), binning=CustomBins()
    )
    values = MonteCarloLimitDistributionResolver(3).resolve(criterion, 20)
    with pytest.raises(TypeError, match="to_config"):
        LimitDistributionModel.from_statistic(criterion, 20, values)
