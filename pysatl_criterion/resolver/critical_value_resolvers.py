from abc import ABC

import numpy as np
import scipy

from pysatl_criterion import IStatistic
from pysatl_criterion.hypothesis import HypothesisType, IHypothesis
from pysatl_criterion.resolver.model import CriticalValueType, ICriticalValueResolver


class AbstractCriticalValueResolver(ICriticalValueResolver, ABC):
    def __init__(self, statistic: IStatistic, hypothesis: IHypothesis):
        self._statistic = statistic
        self._hypothesis = hypothesis

    def _calculate(self, distribution, significance_level: float) -> CriticalValueType:
        distribution.sort()
        ecdf = scipy.stats.ecdf(distribution)
        hypothesis = self._hypothesis

        if hypothesis.hypothesis_type is HypothesisType.TWO_SIDED:
            lower_critical = float(np.quantile(ecdf.cdf.quantiles, q=significance_level / 2))
            upper_critical = float(np.quantile(ecdf.cdf.quantiles, q=1 - significance_level / 2))
            return lower_critical, upper_critical
        elif hypothesis.hypothesis_type is HypothesisType.LEFT:
            critical_value = float(np.quantile(ecdf.cdf.quantiles, q=significance_level))
            return critical_value
        elif hypothesis.hypothesis_type is HypothesisType.RIGHT:
            critical_value = float(np.quantile(ecdf.cdf.quantiles, q=1 - significance_level))
            return critical_value
        else:
            raise ValueError(f" Unknown hypothesis type ${hypothesis.hypothesis_type}")


class StorageCriticalValueResolver(AbstractCriticalValueResolver):
    def __init__(self, statistic: IStatistic, hypothesis: IHypothesis, store):
        super().__init__(statistic, hypothesis)
        self.__store = store

    def resolve(self, size: int, significance_level: float) -> CriticalValueType | None:
        """
         Get critical value from storage.

        :param size: sample size
        :param significance_level:  significance level
        :return: critical value or None
        """
        statistic_code = self._statistic.code()

        critical_value = self.__store.get_critical_value(statistic_code, size, significance_level)
        if critical_value is not None:
            return critical_value

        distribution = self.__store.get_distribution(statistic_code, size)
        if distribution is not None:
            return self._calculate(distribution, significance_level)

        return None


class KnownDistributionCriticalValueResolver(AbstractCriticalValueResolver):
    def __init__(self, statistic: IStatistic, hypothesis: IHypothesis):
        super().__init__(statistic, hypothesis)

    def resolve(self, size: int, significance_level: float) -> CriticalValueType | None:
        """
         Get critical value from storage.

        :param size: sample size
        :param significance_level:  significance level
        :return: critical value or None
        """
        statistic_code = self._statistic.code()
        ks_codes = [
            "KS_NORMALITY_GOODNESS_OF_FIT",
            "KS_EXPONENTIALITY_GOODNESS_OF_FIT",
            "KS_WEIBULL_GOODNESS_OF_FIT",
        ]
        if any(statistic_code in s for s in ks_codes):
            return scipy.stats.distributions.kstwo.ppf(1 - significance_level, size)

        return None


class MonteCarloCriticalValueResolver(AbstractCriticalValueResolver):
    def __init__(self, statistic: IStatistic, hypothesis: IHypothesis, count: int):
        super().__init__(statistic, hypothesis)
        self.__count = count

    def resolve(self, size: int | list[float], significance_level: float) -> CriticalValueType:
        """
         Calculate critical value by Monte Carlo method.

        :param size: sample size
        :param significance_level: significance level
        :return: calculated critical value
        """
        distribution = np.zeros(self.__count)
        hypothesis = self._hypothesis

        for i in range(self.__count):
            x = hypothesis.generate(size=size)
            if type(x) is tuple:
                distribution[i] = self._statistic.execute_statistic(*x)
            else:
                distribution[i] = self._statistic.execute_statistic(x)

        return self._calculate(distribution, significance_level)


class CompositeCriticalValueResolver(ICriticalValueResolver):
    def __init__(self, resolvers: list[ICriticalValueResolver]):
        self.__resolvers = resolvers

    def resolve(self, size: int, significance_level: float) -> CriticalValueType | None:
        """
        Return first non None critical value from list of critical value resolvers.

        :param size: sample size
        :param significance_level: significance level
        :return: first non None critical value
        """
        for resolver in self.__resolvers:
            value = resolver.resolve(size, significance_level)

            if value is not None:
                return value

        return None


class DefaultCriticalValueResolver(CompositeCriticalValueResolver):
    def __init__(
        self,
        statistic: IStatistic,
        hypothesis: IHypothesis,
    ):
        resolvers = [
            KnownDistributionCriticalValueResolver(statistic, hypothesis),
            MonteCarloCriticalValueResolver(statistic, hypothesis, 100_000),
        ]
        super().__init__(resolvers)
