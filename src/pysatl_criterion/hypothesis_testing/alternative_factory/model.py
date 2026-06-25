from abc import ABC, abstractmethod
from typing import Generic

from pysatl_criterion.hypothesis_testing.critical_values.calculator.model import (
    CriticalValueCalculator,
)
from pysatl_criterion.hypothesis_testing.critical_values.critical_area.model import CriticalArea
from pysatl_criterion.hypothesis_testing.model import CriticalValueT
from pysatl_criterion.hypothesis_testing.p_value.calculator.model import PValueCalculator


class AlternativeFactory(ABC, Generic[CriticalValueT]):
    """
    Abstract factory for hypothesis test components tied to an alternative type.
    """

    @abstractmethod
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        """
        Get a critical value calculator for the alternative type.

        :return: critical value calculator.
        """
        pass

    @abstractmethod
    def get_p_value_calculator(self) -> PValueCalculator:
        """
        Get a p-value calculator for the alternative type.

        :return: p-value calculator.
        """
        pass

    @abstractmethod
    def get_critical_area(self, critical_value: CriticalValueT) -> CriticalArea:
        """
        Build a critical area from the provided critical value.

        :param critical_value: critical value or values for the alternative type.
        :return: critical area.
        """
        pass
