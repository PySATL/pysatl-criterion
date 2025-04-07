from abc import ABC, abstractmethod
from collections import namedtuple

from typing_extensions import override

from pysatl_criterion import IStatistic
from pysatl_criterion.gof.goodness_of_fit import AbstractGoodnessOfFitStatistic
from pysatl_criterion.hypothesis import (
    AbstractGofHypothesis,
    AbstractUniformHypothesis,
    HypothesisType,
    IHypothesis,
)
from pysatl_criterion.resolver.critical_value_resolvers import (
    DefaultCriticalValueResolver,
    ICriticalValueResolver,
)
from pysatl_criterion.uniformity.model import AbstractUniformityStatistic


CriterionResult = namedtuple(
    "CriterionResult", ["statistic", "result", "critical_value", "statistic_code"]
)


class ICriterion(ABC):
    hypothesis: IHypothesis

    def __init__(
        self,
        statistic: IStatistic,
        hypothesis: IHypothesis,
        resolver: ICriticalValueResolver | None = None,
    ):
        self.hypothesis = hypothesis
        self._statistic = statistic
        _resolver = resolver
        if _resolver is None:
            _resolver = DefaultCriticalValueResolver(statistic, hypothesis)
        self._resolver = _resolver

    @abstractmethod
    def test(self, significance_level: float, *args, **kwargs) -> CriterionResult:
        """
         Test hypothesis. Calculates statistic test statistics and make a decision to reject or
         not to reject hypothesis.

        :param significance_level: significance level, type I error
        :param args: rvs data to calculated statistic value
        :param kwargs: arguments for statistic calculation
        :return criterion result
        """
        raise NotImplementedError("Method is not implemented")


class AbstractCriterion(ICriterion, ABC):
    def _check_statistic(self, statistic, critical_value):
        if self.hypothesis.hypothesis_type is HypothesisType.TWO_SIDED:
            lower_critical, upper_critical = critical_value
            return lower_critical < statistic < upper_critical
        elif self.hypothesis.hypothesis_type is HypothesisType.LEFT:
            return statistic > critical_value
        elif self.hypothesis.hypothesis_type is HypothesisType.RIGHT:
            return statistic < critical_value
        else:
            raise ValueError(f" Unknown hypothesis type ${self.hypothesis.hypothesis_type}")


class GofCriterion(AbstractCriterion):
    hypothesis: IHypothesis

    def __init__(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        hypothesis: AbstractGofHypothesis,
        resolver: ICriticalValueResolver | None = None,
    ):
        super().__init__(statistic, hypothesis, resolver)

    @override
    def test(self, significance_level: float, x, **kwargs) -> CriterionResult:
        if not (0 < significance_level < 1):
            raise ValueError("Significance level must be greater 0 and lower 1")

        statistic = self._statistic.execute_statistic(x, **kwargs)
        critical_value = self._resolver.resolve(len(x), significance_level)
        result = self._check_statistic(statistic, critical_value)

        return CriterionResult(statistic, result, critical_value, self._statistic.code())


class UniformityCriterion(AbstractCriterion):
    hypothesis: IHypothesis

    def __init__(
        self,
        statistic: AbstractUniformityStatistic,
        hypothesis: AbstractUniformHypothesis,
        resolver: ICriticalValueResolver | None = None,
    ):
        super().__init__(statistic, hypothesis, resolver)

    @override
    def test(self, significance_level: float, x, y, **kwargs) -> CriterionResult:
        if not (0 < significance_level < 1):
            raise ValueError("Significance level must be greater 0 and lower 1")

        statistic = self._statistic.execute_statistic(x, y, **kwargs)
        critical_value = self._resolver.resolve([len(x), len(x)], significance_level)
        result = self._check_statistic(statistic, critical_value)

        return CriterionResult(statistic, result, critical_value, self._statistic.code())
