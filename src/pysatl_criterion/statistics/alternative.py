from abc import ABC, abstractmethod
from enum import Enum

from typing_extensions import override


class AlternativeType(Enum):
    """
    Alternatives for hypotheses.
    """

    RIGHT = "right"
    LEFT = "left"
    TWO_TAILED = "two_tailed"


class Alternative(ABC):
    @staticmethod
    @abstractmethod
    def type() -> AlternativeType:
        raise NotImplementedError("Method is not implemented")

    @abstractmethod
    def check(self, statistic_value: float, cv: float) -> bool:
        raise NotImplementedError("Method is not implemented")

    @classmethod
    def get_alternative(cls, alternative_type: AlternativeType) -> "Alternative":
        if alternative_type == AlternativeType.LEFT:
            return LeftAlternative()
        elif alternative_type == AlternativeType.RIGHT:
            return RightAlternative()
        elif alternative_type == AlternativeType.TWO_TAILED:
            return TwoSidedAlternative()
        else:
            raise ValueError("alternative must be 'LEFT',  'RIGHT' or 'TWO_TAILED'")


class RightAlternative(Alternative):
    @staticmethod
    @override
    def type() -> AlternativeType:
        return AlternativeType.RIGHT

    @override
    def check(self, statistic_value: float, cv: float) -> bool:
        return statistic_value <= cv


class LeftAlternative(Alternative):
    @staticmethod
    @override
    def type() -> AlternativeType:
        return AlternativeType.LEFT

    @override
    def check(self, statistic_value: float, cv: float) -> bool:
        return statistic_value >= cv


class TwoSidedAlternative(Alternative):
    @staticmethod
    @override
    def type() -> AlternativeType:
        return AlternativeType.TWO_TAILED

    @override
    def check(self, statistic_value: float, cv: tuple[float, float]) -> bool:
        if not isinstance(cv, tuple):
            raise TypeError("For a TWO_SIDED hypothesis, 'cv' must be a tuple of two floats.")
        left_cv, right_cv = cv
        return left_cv <= statistic_value <= right_cv
