from abc import ABC

import numpy as np
import scipy

from pysatl_criterion import IStatistic
from pysatl_criterion.hypothesis import IHypothesis
from pysatl_criterion.resolver.model import IPValueResolver


class AbstractPValueResolver(IPValueResolver, ABC):
    def __init__(self, statistic: IStatistic, hypothesis: IHypothesis):
        self._statistic = statistic
        self._hypothesis = hypothesis

    def _calculate(self, distribution, statistic: float) -> float:
        distribution.sort()
        ecdf = scipy.stats.ecdf(distribution)

        return np.take(ecdf.sf.evaluate(np.array([statistic])), 0)


class MonteCarloPValueResolver(AbstractPValueResolver):
    def __init__(self, statistic: IStatistic, hypothesis: IHypothesis, count: int):
        super().__init__(statistic, hypothesis)
        self.__count = count

    def resolve(self, x: list[float]) -> float:
        """
         P-value by Monte Carlo method.

        :param x: sample
        :return: calculated p-value
        """
        distribution = np.zeros(self.__count)
        hypothesis = self._hypothesis
        size = len(x)

        for i in range(self.__count):
            x = hypothesis.generate(size=size)
            if type(x) is tuple:
                distribution[i] = self._statistic.execute_statistic(*x)
            else:
                distribution[i] = self._statistic.execute_statistic(x)

        statistic = self._statistic.execute_statistic(x)
        print("statistic", statistic)
        return self._calculate(distribution, statistic)
