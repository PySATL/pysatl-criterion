from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, overload

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

    @overload
    @classmethod
    def get_alternative(
        cls, alternative_type: Literal[AlternativeType.LEFT]
    ) -> "LeftAlternative": ...

    @overload
    @classmethod
    def get_alternative(
        cls, alternative_type: Literal[AlternativeType.RIGHT]
    ) -> "RightAlternative": ...

    @overload
    @classmethod
    def get_alternative(
        cls, alternative_type: Literal[AlternativeType.TWO_TAILED]
    ) -> "TwoSidedAlternative": ...

    @overload
    @classmethod
    def get_alternative(cls, alternative_type: AlternativeType) -> "Alternative": ...

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


class LeftAlternative(Alternative):
    @staticmethod
    @override
    def type() -> AlternativeType:
        return AlternativeType.LEFT


class TwoSidedAlternative(Alternative):
    @staticmethod
    @override
    def type() -> AlternativeType:
        return AlternativeType.TWO_TAILED
