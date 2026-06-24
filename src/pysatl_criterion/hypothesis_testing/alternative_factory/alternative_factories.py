from statistics.alternative import AlternativeType

from hypothesis_testing.alternative_factory.model import AlternativeFactory
from hypothesis_testing.critical_values.calculator.critical_value_calculator import (
    LeftCriticalValueCalculator,
    RightCriticalValueCalculator,
    TwoSidedCriticalValueCalculator,
)
from hypothesis_testing.critical_values.critical_area.critical_areas import (
    LeftCriticalArea,
    RightCriticalArea,
    TwoSidedCriticalArea,
)
from hypothesis_testing.critical_values.critical_area.model import CriticalArea
from hypothesis_testing.p_value.calculator.p_value_calculator import (
    LeftPValueCalculator,
    RightPValueCalculator,
    TwoSidedPValueCalculator,
)
from typing_extensions import override

from pysatl_criterion import CriticalValueCalculator, PValueCalculator


class LeftAlternativeFactory(AlternativeFactory[float]):
    @override
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        return LeftCriticalValueCalculator()

    @override
    def get_p_value_calculator(self) -> PValueCalculator:
        return LeftPValueCalculator()

    @override
    def get_critical_area(self, critical_value: float) -> CriticalArea:
        return LeftCriticalArea(critical_value)


class RightAlternativeFactory(AlternativeFactory[float]):
    @override
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        return RightCriticalValueCalculator()

    @override
    def get_p_value_calculator(self) -> PValueCalculator:
        return RightPValueCalculator()

    @override
    def get_critical_area(self, critical_value: float) -> CriticalArea:
        return RightCriticalArea(critical_value)


class TwoSidedAlternativeFactory(AlternativeFactory[tuple[float, float]]):
    @override
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        return TwoSidedCriticalValueCalculator()

    @override
    def get_p_value_calculator(self) -> PValueCalculator:
        return TwoSidedPValueCalculator()

    @override
    def get_critical_area(self, critical_value: tuple[float, float]) -> CriticalArea:
        left_critical_value, right_critical_value = critical_value
        return TwoSidedCriticalArea(left_critical_value, right_critical_value)


class AbstractAlternativeFactory:
    @staticmethod
    def get_concrete_factory(
        alternative_type: AlternativeType,
    ) -> AlternativeFactory[float] | AlternativeFactory[tuple[float, float]]:
        if alternative_type == AlternativeType.LEFT:
            return LeftAlternativeFactory()
        elif alternative_type == AlternativeType.RIGHT:
            return RightAlternativeFactory()
        else:
            return TwoSidedAlternativeFactory()
