import importlib
import importlib.metadata
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import pysatl_criterion
from pysatl_criterion import DistributionType
from pysatl_criterion.distribution import (
    DistributionDescriptor,
)
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
from pysatl_criterion.hypothesis_testing.critical_values.calculator.model import (
    CriticalValueCalculator,
)
from pysatl_criterion.hypothesis_testing.critical_values.critical_area.model import CriticalArea
from pysatl_criterion.hypothesis_testing.goodness_of_fit_test import goodness_of_fit_test
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    AbstractLimitDistributionResolver,
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.hypothesis_testing.model import DecisionMethod
from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator
from pysatl_criterion.multiple_testing import (
    BenjaminiYekutieli,
    Holm,
    HolmSidak,
)
from pysatl_criterion.persistence.sqlalchemy.types import CompressedFloatArray
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
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
from pysatl_criterion.statistics.goodness_of_fit.edf import (
    AbstractAndersonDarlingStatistic,
    AbstractCramerVonMisesStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis
from pysatl_criterion.statistics.independence.independence_statistic import (
    AbstractIndependenceStatistic,
)
from pysatl_criterion.statistics.statistic import (
    AbstractStatistic,
)
from tests.parameter_cases import parameters_for


@pytest.mark.parametrize(
    "base",
    [
        AbstractStatistic,
        AbstractGoodnessOfFitStatistic,
        AbstractIndependenceStatistic,
        DistributionDescriptor,
        AbstractLimitDistributionResolver,
        DecisionMethod,
        CriticalValueCalculator,
        PValueCalculator,
        CriticalArea,
    ],
)
def test_abstract_interfaces_cannot_be_instantiated(base):
    with pytest.raises(TypeError, match="abstract"):
        base()


def test_base_family_identifiers_and_empty_hypothesis():
    assert AbstractGoodnessOfFitStatistic.code() == "GOODNESS_OF_FIT"
    assert AbstractIndependenceStatistic.code() == "INDEPENDENCE"
    assert GoodnessOfFitHypothesis(None).parameters() == {}


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
                statistic = statistic_class(parameters_for(statistic_class))
                assert statistic.code()
                assert statistic.short_code()
                assert statistic.alternative()
                assert issubclass(statistic.distribution(), DistributionDescriptor)
                assert not inspect.isabstract(statistic.distribution())
                assert isinstance(statistic.distribution().type(), DistributionType)
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
        "pysatl_criterion.hypothesis_testing.limit_distribution.base.get_hypothesis_generator",
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
    assert AbstractAndersonDarlingStatistic.short_code() == "AD"
    assert AbstractCramerVonMisesStatistic.short_code() == "CVM"
    assert Holm().adjust(np.array([])).size == 0
    assert HolmSidak().adjust(np.array([])).size == 0
    assert BenjaminiYekutieli().adjust(np.array([])).size == 0
    assert len(BenjaminiYekutieli().adjust(np.full(10001, 0.5))) == 10001
    assert BenjaminiYekutieli().adjust(np.array([0.4, 0.01])) == pytest.approx([0.6, 0.03])
    assert BenjaminiYekutieli().adjust(np.array([0.4, 0.41])) == pytest.approx([0.615, 0.615])


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
