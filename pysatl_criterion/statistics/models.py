from abc import ABC, abstractmethod
from enum import Enum, auto

from numpy import float64


class HypothesisType(Enum):
    """
    Alternatives for hypotheses.
    """

    RIGHT = auto()
    LEFT = auto()
    TWO_TAILED = auto()

    def check_hypothesis(
        self,
        statistic_value,
        cv: float | tuple[float, float],
    ) -> bool:
        """
        Compares the value of a statistic to a critical value
        param statistic_value: statistic value
        param cv: critical value

        return: True if hypothesis is valid, False otherwise

        """
        if self == HypothesisType.RIGHT:
            return statistic_value <= cv
        if self == HypothesisType.LEFT:
            return statistic_value >= cv
        if self == HypothesisType.TWO_TAILED:
            if not isinstance(cv, tuple):
                raise TypeError("For a TWO_SIDED hypothesis, 'cv' must be a tuple of two floats.")
            left_cv, right_cv = cv
            return left_cv <= statistic_value <= right_cv


class AbstractStatistic(ABC):
    @staticmethod
    @abstractmethod
    def code() -> str:
        """
        Generate unique code for test statistic.
        """
        raise NotImplementedError("Method is not implemented")

    @abstractmethod
    def execute_statistic(self, rvs, **kwargs) -> float | float64:
        """
        Execute test statistic and return calculated statistic value.
        :param rvs: rvs data to calculated statistic value
        :param kwargs: arguments for statistic calculation
        """
        raise NotImplementedError("Method is not implemented")
