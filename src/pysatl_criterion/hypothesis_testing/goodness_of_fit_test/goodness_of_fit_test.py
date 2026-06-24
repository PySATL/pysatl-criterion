from typing_extensions import override

from pysatl_criterion.hypothesis_testing.alternative_factory.alternative_factories import (
    AbstractAlternativeFactory,
)
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
    """
    Decision method that rejects a hypothesis by comparing p-value with significance level.
    """

    def __init__(self, resolver: AbstractLimitDistributionResolver | None = None):
        """
        Initialize the p-value decision method.

        :param resolver: limit distribution resolver. Uses Monte Carlo resolver by default.
        """
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
        """
        Decide whether to reject the hypothesis using a p-value.

        :param statistic: goodness-of-fit statistic definition.
        :param statistic_value: computed statistic value for the observed sample.
        :param significance_level: test significance level.
        :param sample_size: observed sample size.
        :return: test result with p-value and rejection flag.
        """
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
    """
    Decision method that rejects a hypothesis using critical values and a critical area.
    """

    def __init__(self, resolver: AbstractLimitDistributionResolver | None = None):
        """
        Initialize the critical value decision method.

        :param resolver: limit distribution resolver. Uses Monte Carlo resolver by default.
        """
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
        """
        Decide whether to reject the hypothesis using critical values.

        :param statistic: goodness-of-fit statistic definition.
        :param statistic_value: computed statistic value for the observed sample.
        :param significance_level: test significance level.
        :param sample_size: observed sample size.
        :return: test result with critical value and rejection flag.
        """
        limit_distribution = self.resolver.resolve(statistic, sample_size)
        factory = AbstractAlternativeFactory.get_concrete_factory(statistic.alternative().type())
        calculator = factory.get_critical_value_calculator()
        critical_value = calculator.calculate(limit_distribution, significance_level)
        region = factory.get_critical_area(critical_value)

        return TestResult(
            statistic=statistic_value,
            significance_level=significance_level,
            p_value=None,
            critical_value=critical_value,
            rejected=not region.contains(statistic_value),
        )


class GoodnessOfFitTest:
    """
    Goodness-of-fit test runner for a configured statistic.
    """

    def __init__(self, statistic: AbstractGoodnessOfFitStatistic):
        """
        Initialize the goodness-of-fit test.

        :param statistic: statistic used to evaluate observed samples.
        """
        self.statistic = statistic

    def test(
        self,
        rvs: list[float],
        significance_level: float,
        method: DecisionMethod | None = None,
    ) -> TestResult:
        """
        Run the goodness-of-fit test on observed values.

        :param rvs: observed random values.
        :param significance_level: test significance level.
        :param method: decision method. Uses critical value method by default.
        :return: goodness-of-fit test result.
        """
        if method is None:
            method = CriticalValueDecisionMethod()

        return method.decide(
            self.statistic, self.statistic.execute_statistic(rvs), significance_level, len(rvs)
        )
