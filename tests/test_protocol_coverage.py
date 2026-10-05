import importlib
import importlib.metadata
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import pysatl_criterion
from pysatl_criterion import DistributionType
from pysatl_criterion.distribution.distributions import DistributionDescriptor
from pysatl_criterion.estimation.maximum_likelihood import (
    function_for_maximum_likelihood as likelihoods,
)
from pysatl_criterion.estimation.maximum_likelihood import maximum_likelihood
from pysatl_criterion.hypothesis_testing.alternative_factory.alternative_factories import (
    AbstractAlternativeFactory,
    LeftAlternativeFactory,
    RightAlternativeFactory,
    TwoSidedAlternativeFactory,
)
from pysatl_criterion.hypothesis_testing.alternative_factory.model import AlternativeFactory
from pysatl_criterion.hypothesis_testing.critical_values.calculator.model import (
    CriticalValueCalculator,
)
from pysatl_criterion.hypothesis_testing.critical_values.critical_area.model import CriticalArea
from pysatl_criterion.hypothesis_testing.critical_values.loader.remote_loader import (
    CriticalValueLoader,
)
from pysatl_criterion.hypothesis_testing.goodness_of_fit_test import goodness_of_fit_test
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    AbstractLimitDistributionResolver,
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.hypothesis_testing.model import DecisionMethod
from pysatl_criterion.hypothesis_testing.multiple_testing.abstract_multiple_testing import (
    AbstractMultipleTesting,
)
from pysatl_criterion.hypothesis_testing.multiple_testing.fdr import BenjaminiYekutieli
from pysatl_criterion.hypothesis_testing.multiple_testing.fwer import Holm, SidakHolm
from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator
from pysatl_criterion.persistence.models.base import IDataStorage, IStorage
from pysatl_criterion.persistence.models.limit_distribution import ILimitDistributionStorage
from pysatl_criterion.persistence.sqlalchemy.alchemy_decorator import CompressedFloatArray
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType
from pysatl_criterion.statistics.goodness_of_fit import (
    beta,
    exponent,
    gamma,
    inverse_gamma,
    log_logistic,
    log_normal,
    pareto,
    student,
    uniform,
)
from pysatl_criterion.statistics.goodness_of_fit.common import (
    ADStatistic,
    Chi2Statistic,
    CrammerVonMisesStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis
from pysatl_criterion.statistics.statistic import (
    AbstractGoodnessOfFitStatistic,
    AbstractIndependenceStatistic,
    AbstractStatistic,
)


def test_abstract_protocol_method_bodies():
    assert DistributionDescriptor.type() is None
    assert DistributionDescriptor.parameters() is None
    assert AlternativeFactory.get_critical_value_calculator(object()) is None
    assert AlternativeFactory.get_p_value_calculator(object()) is None
    assert AlternativeFactory.get_critical_area(object(), 1) is None
    assert CriticalValueCalculator.calculate(object(), [], 0.05) is None
    assert CriticalArea.contains(object(), 1) is None
    assert PValueCalculator.calculate(object(), [], 1) is None
    assert IStorage.init(object()) is None
    assert IDataStorage.get_data(object(), object()) is None
    assert IDataStorage.insert_data(object(), object()) is None
    assert IDataStorage.delete_data(object(), object()) is None
    assert ILimitDistributionStorage.get_data_for_cv(object(), object()) is None
    assert ILimitDistributionStorage.get_bulk_data(object(), [], 1) is None
    assert ILimitDistributionStorage.insert_bulk_data(object(), []) is None
    assert DecisionMethod.decide(object(), object(), 1, 0.05, 10) is None
    assert GoodnessOfFitHypothesis(None).parameters() == []
    assert AbstractStatistic.hypothesis(object()) is None
    assert AbstractStatistic.alternative(object()) is None
    assert AbstractGoodnessOfFitStatistic.hypothesis(object()) is None
    assert AbstractGoodnessOfFitStatistic.distribution() is None
    assert AbstractIndependenceStatistic.hypothesis(object()) is None
    assert AbstractGoodnessOfFitStatistic.code() == "GOODNESS_OF_FIT"
    assert AbstractIndependenceStatistic.code() == "INDEPENDENCE"

    for method, args in (
        (AbstractStatistic.code, ()),
        (AbstractStatistic.short_code, ()),
        (AbstractGoodnessOfFitStatistic.execute_statistic, (object(), [])),
        (AbstractIndependenceStatistic.execute_statistic, (object(), [], [])),
        (AbstractLimitDistributionResolver.resolve, (object(), object(), 1)),
        (AbstractMultipleTesting.adjust, ([0.1],)),
        (Alternative.type, ()),
    ):
        with pytest.raises(NotImplementedError):
            method(*args)


@pytest.mark.parametrize(
    ("alternative_type", "factory_type", "critical_value"),
    [
        (AlternativeType.LEFT, LeftAlternativeFactory, 1.0),
        (AlternativeType.RIGHT, RightAlternativeFactory, 1.0),
        (AlternativeType.TWO_TAILED, TwoSidedAlternativeFactory, (1.0, 2.0)),
    ],
)
def test_alternative_factories(alternative_type, factory_type, critical_value):
    factory = AbstractAlternativeFactory.get_concrete_factory(alternative_type)
    assert isinstance(factory, factory_type)
    assert factory.get_critical_value_calculator()
    assert factory.get_p_value_calculator()
    assert factory.get_critical_area(critical_value)


def test_remote_loader_single_item_paths():
    local = Mock()
    assert not CriticalValueLoader(local, None).load("criterion", 10)

    remote = Mock()
    remote.get_data_for_cv.return_value = SimpleNamespace(criterion_code="criterion")
    assert CriticalValueLoader(local, remote).load("criterion", 10)

    remote.get_data_for_cv.side_effect = [None, SimpleNamespace(criterion_code="criterion")]
    assert CriticalValueLoader(local, remote).load("criterion", 10)
    local.insert_data.assert_called()

    remote.get_data_for_cv.side_effect = None
    remote.get_data_for_cv.return_value = None
    assert not CriticalValueLoader(local, remote).load("missing", 10)


def test_remote_loader_bulk_edge_paths():
    local = Mock()
    remote = Mock()
    assert CriticalValueLoader(local, remote).load_bulk([], 10).requested_count == 0
    assert CriticalValueLoader(local, None).load_bulk(["a"], 10).not_found_codes == ["a"]

    local.get_data_for_cv.return_value = True
    assert CriticalValueLoader(local, remote).load_bulk(["a"], 10).already_cached_count == 1

    local.get_data_for_cv.side_effect = [True, False]
    remote.get_bulk_data.return_value = [SimpleNamespace(criterion_code="b")]
    local.insert_bulk_data.side_effect = RuntimeError("write failed")
    result = CriticalValueLoader(local, remote).load_bulk(["a", "b"], 10)
    assert result.newly_cached_count == 1
    assert result.not_found_codes == []


def test_all_goodness_of_fit_classes_expose_the_common_contract():
    modules = [
        beta,
        exponent,
        gamma,
        inverse_gamma,
        log_logistic,
        log_normal,
        pareto,
        student,
        uniform,
    ]
    count = 0
    for module in modules:
        for _, statistic_class in inspect.getmembers(module, inspect.isclass):
            if (
                statistic_class.__module__ == module.__name__
                and issubclass(statistic_class, AbstractGoodnessOfFitStatistic)
                and not inspect.isabstract(statistic_class)
            ):
                statistic = statistic_class()
                assert statistic.code()
                assert statistic.short_code()
                assert statistic.alternative()
                assert isinstance(statistic.distribution(), DistributionType)
                assert statistic.hypothesis()
                count += 1
    assert count > 100


def test_monte_carlo_resolver_validates_constructor_arguments():
    with pytest.raises(ValueError):
        MonteCarloLimitDistributionResolver(0)
    resolver = MonteCarloLimitDistributionResolver(1)
    with pytest.raises(ValueError, match="sample_size"):
        resolver.resolve(Mock(), 0)


def test_monte_carlo_resolver_simulates_values(monkeypatch):
    statistic = Mock()
    statistic.distribution.return_value = DistributionType.NORMAL
    statistic.hypothesis.return_value = SimpleNamespace(params={"mean": 0, "var": 1})
    statistic.execute_statistic.side_effect = lambda values: values[0]
    generator = Mock()
    generator.generate.return_value = [2.5]
    monkeypatch.setattr(
        "pysatl_criterion.hypothesis_testing.limit_distribution.base.get_available_generator",
        lambda *_args: generator,
    )
    assert MonteCarloLimitDistributionResolver(2).resolve(statistic, 1) == [2.5, 2.5]


def test_decision_method_defaults_and_missing_distribution_errors():
    resolver = Mock()
    resolver.resolve.return_value = None
    statistic = Mock()
    statistic.alternative.return_value.type.return_value = AlternativeType.RIGHT

    assert goodness_of_fit_test.PValueDecisionMethod().resolver
    assert goodness_of_fit_test.CriticalValueDecisionMethod().resolver
    with pytest.raises(ValueError, match="cannot be None"):
        goodness_of_fit_test.PValueDecisionMethod(resolver).decide(statistic, 1, 0.05, 10)
    with pytest.raises(ValueError, match="cannot be None"):
        goodness_of_fit_test.CriticalValueDecisionMethod(resolver).decide(statistic, 1, 0.05, 10)


def test_common_statistic_helpers_and_empty_multiple_testing_inputs():
    with pytest.raises(NotImplementedError):
        ADStatistic.code()
    masked = np.ma.array([1.0, 2.0], mask=[False, True])
    assert Chi2Statistic._m_sum(masked, axis=0, preserve_mask=True, xp=np) == 1
    assert Chi2Statistic._m_sum(masked, axis=0, preserve_mask=False, xp=np) == 1
    assert Chi2Statistic._m_sum(np.array([1.0, 2.0]), axis=0, preserve_mask=False, xp=np) == 3
    assert np.isfinite(Chi2Statistic.do_execute_statistic(object(), [2, 3], np.array([1, 2]), -1))
    assert CrammerVonMisesStatistic.code() == "CVM"
    assert Holm.adjust([]) == []
    assert SidakHolm.adjust([]) == []
    assert BenjaminiYekutieli.adjust([]) == []
    assert len(BenjaminiYekutieli.adjust([0.5] * 10001)) == 10001
    assert BenjaminiYekutieli.adjust([0.4, 0.01]) == pytest.approx([0.6, 0.03])
    assert BenjaminiYekutieli.adjust([0.4, 0.41]) == pytest.approx([0.615, 0.615])


def test_compressed_float_array_compatibility_and_errors():
    value_type = CompressedFloatArray()
    assert value_type.process_bind_param(None, None) is None
    assert value_type.process_result_value(None, None) is None
    assert value_type.process_result_value("[1.0, 2.0]", None) == [1.0, 2.0]
    packed = value_type.process_bind_param("[1.0, 2.0]", None)
    assert value_type.process_result_value(packed, None) == [1.0, 2.0]

    float32_type = CompressedFloatArray(use_float32=True)
    packed32 = float32_type.process_bind_param([1.0], None)
    assert float32_type.process_result_value(packed32, None) == [1.0]
    with pytest.raises(ValueError, match="too short"):
        value_type.process_result_value(b"bad", None)
    with pytest.raises(ValueError, match="Unsupported version"):
        value_type.process_result_value(bytes([2]) + packed[1:], None)


def test_maximum_likelihood_selection_and_validation(monkeypatch):
    # The automatic search only considers single-sample ``likelihood_function_*``
    # helpers defined in the module; imported optimisers and the binomial
    # likelihood (which needs an explicit number of trials) must be excluded.
    candidates = maximum_likelihood._single_sample_likelihood_functions()
    candidate_names = {function.__name__ for function in candidates}
    assert candidate_names
    assert all(name.startswith("likelihood_function_") for name in candidate_names)
    assert "likelihood_function_binomial" not in candidate_names
    assert "minimize_scalar" not in candidate_names
    assert "root_scalar" not in candidate_names

    data = [0.25, 0.26, 0.23, 0.24, 0.27]
    best = maximum_likelihood.find_max_probability_function(data)
    assert best in candidates

    function, value = maximum_likelihood.find_result_of_likelihood_function(data)
    assert function is best
    assert value > 0

    with pytest.raises(ValueError, match="no data"):
        maximum_likelihood.find_result_of_likelihood_function([])
    with pytest.raises(TypeError, match="numeric"):
        maximum_likelihood.find_result_of_likelihood_function(["x"])

    monkeypatch.setattr(maximum_likelihood, "find_max_probability_function", lambda _data: None)
    with pytest.raises(ValueError, match="No suitable"):
        maximum_likelihood.find_result_of_likelihood_function([1, 2])


def test_likelihood_numerical_fallbacks(monkeypatch):
    assert likelihoods.likelihood_function_weibull(np.array([0.5, 1.0, 2.0, 3.0])) >= 0

    monkeypatch.setattr(likelihoods.cauchy, "fit", lambda _data: (0, 0))
    assert likelihoods.likelihood_function_cauchy(np.array([1.0, 2.0])) == 0

    monkeypatch.setattr(
        likelihoods,
        "minimize_scalar",
        lambda function, **_kwargs: SimpleNamespace(success=False, fun=function(0)),
    )
    assert likelihoods.likelihood_function_wigner(np.array([-1.0, 1.0])) == 0
    assert likelihoods.likelihood_function_binomial(np.array([1, 2, 3]), 5) > 0


def test_cauchy_likelihood_overflow_fallback(monkeypatch):
    monkeypatch.setattr(likelihoods.cauchy, "fit", lambda _data: (0, 1))
    monkeypatch.setattr(likelihoods.cauchy, "logpdf", lambda *_args, **_kwargs: np.array([1.0]))
    monkeypatch.setattr(likelihoods.np, "exp", Mock(side_effect=OverflowError))
    assert likelihoods.likelihood_function_cauchy(np.array([1.0, 2.0])) == float("inf")


def test_package_version_fallback(monkeypatch):
    original_version = pysatl_criterion.__version__
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        Mock(side_effect=importlib.metadata.PackageNotFoundError),
    )
    try:
        assert importlib.reload(pysatl_criterion).__version__ == "0+unknown"
    finally:
        # Restore the real metadata lookup and the module-level state so the
        # rest of the test session is not affected by the reload above.
        monkeypatch.undo()
        importlib.reload(pysatl_criterion)

    assert pysatl_criterion.__version__ == original_version
