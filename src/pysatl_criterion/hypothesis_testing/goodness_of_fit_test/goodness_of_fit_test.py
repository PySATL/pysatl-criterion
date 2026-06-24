from hypothesis_testing.alternative_factory.alternative_factories import AbstractAlternativeFactory
from typing_extensions import override

from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    AbstractLimitDistributionResolver,
    MonteCarloLimitDistributionResolver,
)
from pysatl_criterion.hypothesis_testing.model import (
    DecisionMethod,
    TestResult,
)
from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic


class PValueDecisionMethod(DecisionMethod):
    def __init__(self, resolver: AbstractLimitDistributionResolver | None = None):
        if resolver is None:
            resolver = MonteCarloLimitDistributionResolver(10_000)
        self.resolver = resolver

    @override
    def decide(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        statistic_value: float,
        significance_level: float,
        sample_size: int,
    ) -> TestResult:

        limit_distribution = self.resolver.resolve(statistic, sample_size)
        factory = AbstractAlternativeFactory.get_concrete_factory(statistic.alternative().type())
        calculator = factory.get_p_value_calculator()
        p_value = calculator.calculate(limit_distribution, statistic_value)

        return TestResult(
            statistic=statistic_value,
            significance_level=significance_level,
            p_value=p_value,
            critical_value=None,
            rejected=p_value <= significance_level,
        )


class CriticalValueDecisionMethod(DecisionMethod):
    def __init__(self, resolver: AbstractLimitDistributionResolver | None = None):
        if resolver is None:
            resolver = MonteCarloLimitDistributionResolver(10_000)
        self.resolver = resolver

    @override
    def decide(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        statistic_value: float,
        significance_level: float,
        sample_size: int,
    ) -> TestResult:
        limit_distribution = self.resolver.resolve(statistic, sample_size)
        factory = AbstractAlternativeFactory.get_concrete_factory(statistic.alternative().type())
        calculator = factory.get_critical_value_calculator()
        critical_value = calculator.calculate(limit_distribution, statistic_value)
        region = factory.get_critical_area(critical_value)

        return TestResult(
            statistic=statistic_value,
            significance_level=significance_level,
            p_value=None,
            critical_value=critical_value,
            rejected=not region.contains(statistic_value),
        )


class GoodnessOfFitTest:
    def __init__(self, statistic: AbstractGoodnessOfFitStatistic):
        self.statistic = statistic

    def test(
        self,
        rvs: list[float],
        significance_level: float,
        method: DecisionMethod | None = None,
    ) -> TestResult:
        if method is None:
            method = CriticalValueDecisionMethod()

        return method.decide(
            self.statistic, self.statistic.execute_statistic(rvs), significance_level, len(rvs)
        )
