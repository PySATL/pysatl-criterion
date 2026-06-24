from pysatl_criterion import DistributionType
from pysatl_criterion.hypothesis_testing.goodness_of_fit_test import (
    goodness_of_fit_test as gof_module,
)
from pysatl_criterion.hypothesis_testing.goodness_of_fit_test.goodness_of_fit_test import (
    CriticalValueDecisionMethod,
    GoodnessOfFitTest,
    PValueDecisionMethod,
)
from pysatl_criterion.hypothesis_testing.model import DecisionMethod, TestResult
from pysatl_criterion.statistics.alternative import Alternative, AlternativeType
from pysatl_criterion.statistics.hypothesis import GoodnessOfFitHypothesis


class FakeResolver:
    def __init__(self, limit_distribution):
        self.limit_distribution = limit_distribution
        self.calls = []

    def resolve(self, statistic, sample_size):
        self.calls.append((statistic, sample_size))
        return self.limit_distribution


class FakeStatistic:
    def __init__(self, statistic_value=2.0, alternative_type=AlternativeType.RIGHT):
        self.statistic_value = statistic_value
        self.alternative_type = alternative_type
        self.executed_with = None

    def hypothesis(self):
        return GoodnessOfFitHypothesis(DistributionType.NORMAL, {})

    def alternative(self):
        return Alternative.get_alternative(self.alternative_type)

    @staticmethod
    def distribution():
        return DistributionType.NORMAL

    def execute_statistic(self, rvs):
        self.executed_with = rvs
        return self.statistic_value


class RecordingDecisionMethod(DecisionMethod):
    def __init__(self):
        self.calls = []

    def decide(self, statistic, statistic_value, significance_level, sample_size):
        self.calls.append((statistic, statistic_value, significance_level, sample_size))
        return TestResult(
            statistic=statistic_value,
            significance_level=significance_level,
            p_value=None,
            critical_value=10.0,
            rejected=False,
        )


def test_goodness_of_fit_test_delegates_to_supplied_decision_method():
    statistic = FakeStatistic(statistic_value=3.5)
    method = RecordingDecisionMethod()
    rvs = [1.0, 2.0, 3.0]

    result = GoodnessOfFitTest(statistic).test(rvs, 0.05, method)

    assert statistic.executed_with == rvs
    assert method.calls == [(statistic, 3.5, 0.05, 3)]
    assert result == TestResult(
        statistic=3.5,
        significance_level=0.05,
        p_value=None,
        critical_value=10.0,
        rejected=False,
    )


def test_goodness_of_fit_test_uses_critical_value_method_by_default(monkeypatch):
    created_methods = []

    class FakeDefaultDecisionMethod(RecordingDecisionMethod):
        def __init__(self):
            super().__init__()
            created_methods.append(self)

    monkeypatch.setattr(gof_module, "CriticalValueDecisionMethod", FakeDefaultDecisionMethod)

    statistic = FakeStatistic(statistic_value=4.0)
    result = GoodnessOfFitTest(statistic).test([10.0, 20.0], 0.1)

    assert len(created_methods) == 1
    assert created_methods[0].calls == [(statistic, 4.0, 0.1, 2)]
    assert result.critical_value == 10.0


def test_p_value_decision_method_resolves_distribution_and_rejects_on_boundary():
    statistic = FakeStatistic(statistic_value=2.0, alternative_type=AlternativeType.RIGHT)
    resolver = FakeResolver([1.0, 2.0, 3.0, 4.0])
    method = PValueDecisionMethod(resolver)

    result = method.decide(statistic, 2.0, 0.5, sample_size=4)

    assert resolver.calls == [(statistic, 4)]
    assert result == TestResult(
        statistic=2.0,
        significance_level=0.5,
        p_value=0.5,
        critical_value=None,
        rejected=True,
    )


def test_critical_value_decision_method_uses_significance_level_for_calculation():
    statistic = FakeStatistic(statistic_value=2.5, alternative_type=AlternativeType.RIGHT)
    resolver = FakeResolver([0.0, 1.0, 2.0, 3.0])
    method = CriticalValueDecisionMethod(resolver)

    result = method.decide(statistic, 2.5, 0.25, sample_size=4)

    assert resolver.calls == [(statistic, 4)]
    assert result == TestResult(
        statistic=2.5,
        significance_level=0.25,
        p_value=None,
        critical_value=2.25,
        rejected=True,
    )
