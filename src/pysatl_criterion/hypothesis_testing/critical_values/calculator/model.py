from abc import ABC, abstractmethod
from typing import Generic

from pysatl_criterion.hypothesis_testing.model import CriticalValueT


class CriticalValueCalculator(ABC, Generic[CriticalValueT]):
    """
    Abstract calculator for critical values of hypothesis tests.
    """

    @abstractmethod
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> CriticalValueT:
        """
        Calculate critical value or values from a limit distribution.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param significance_level: test significance level.
        :return: critical value representation for the calculator type.
        """
        pass
