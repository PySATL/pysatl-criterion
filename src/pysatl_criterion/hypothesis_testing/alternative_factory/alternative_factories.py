from typing_extensions import override

from pysatl_criterion.hypothesis_testing.alternative_factory.model import AlternativeFactory
from pysatl_criterion.hypothesis_testing.critical_values.calculator import (
    critical_value_calculator as cv_calculators,
)
from pysatl_criterion.hypothesis_testing.critical_values.calculator.model import (
    CriticalValueCalculator,
)
from pysatl_criterion.hypothesis_testing.critical_values.critical_area.critical_areas import (
    LeftCriticalArea,
    RightCriticalArea,
    TwoSidedCriticalArea,
)
from pysatl_criterion.hypothesis_testing.critical_values.critical_area.model import CriticalArea
from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator
from pysatl_criterion.hypothesis_testing.p_value.calculator.p_value_calculator import (
    LeftPValueCalculator,
    RightPValueCalculator,
    TwoSidedPValueCalculator,
)
from pysatl_criterion.statistics.alternative import AlternativeType


class LeftAlternativeFactory(AlternativeFactory[float]):
    """
    Factory for left-tailed hypothesis test components.
    """

    @override
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        """
        Get a critical value calculator for left-tailed tests.

        :return: left-tailed critical value calculator.
        """
        return cv_calculators.LeftCriticalValueCalculator()

    @override
    def get_p_value_calculator(self) -> PValueCalculator:
        """
        Get a p-value calculator for left-tailed tests.

        :return: left-tailed p-value calculator.
        """
        return LeftPValueCalculator()

    @override
    def get_critical_area(self, critical_value: float) -> CriticalArea:
        """
        Build a critical area for a left-tailed test.

        :param critical_value: boundary value for the left critical area.
        :return: left critical area.
        """
        return LeftCriticalArea(critical_value)


class RightAlternativeFactory(AlternativeFactory[float]):
    """
    Factory for right-tailed hypothesis test components.
    """

    @override
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        """
        Get a critical value calculator for right-tailed tests.

        :return: right-tailed critical value calculator.
        """
        return cv_calculators.RightCriticalValueCalculator()

    @override
    def get_p_value_calculator(self) -> PValueCalculator:
        """
        Get a p-value calculator for right-tailed tests.

        :return: right-tailed p-value calculator.
        """
        return RightPValueCalculator()

    @override
    def get_critical_area(self, critical_value: float) -> CriticalArea:
        """
        Build a critical area for a right-tailed test.

        :param critical_value: boundary value for the right critical area.
        :return: right critical area.
        """
        return RightCriticalArea(critical_value)


class TwoSidedAlternativeFactory(AlternativeFactory[tuple[float, float]]):
    """
    Factory for two-sided hypothesis test components.
    """

    @override
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        """
        Get a critical value calculator for two-sided tests.

        :return: two-sided critical value calculator.
        """
        return cv_calculators.TwoSidedCriticalValueCalculator()

    @override
    def get_p_value_calculator(self) -> PValueCalculator:
        """
        Get a p-value calculator for two-sided tests.

        :return: two-sided p-value calculator.
        """
        return TwoSidedPValueCalculator()

    @override
    def get_critical_area(self, critical_value: tuple[float, float]) -> CriticalArea:
        """
        Build a critical area for a two-sided test.

        :param critical_value: left and right boundary values for the critical area.
        :return: two-sided critical area.
        """
        left_critical_value, right_critical_value = critical_value
        return TwoSidedCriticalArea(left_critical_value, right_critical_value)


class AbstractAlternativeFactory:
    """
    Selector for concrete factories by alternative type.
    """

    @staticmethod
    def get_concrete_factory(
        alternative_type: AlternativeType,
    ) -> AlternativeFactory[float] | AlternativeFactory[tuple[float, float]]:
        """
        Get a concrete alternative factory for the provided alternative type.

        :param alternative_type: type of hypothesis alternative.
        :return: alternative factory matching the requested alternative type.
        """
        if alternative_type == AlternativeType.LEFT:
            return LeftAlternativeFactory()
        elif alternative_type == AlternativeType.RIGHT:
            return RightAlternativeFactory()
        else:
            return TwoSidedAlternativeFactory()
