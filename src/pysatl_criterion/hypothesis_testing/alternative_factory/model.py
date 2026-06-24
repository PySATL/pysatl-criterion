from abc import ABC, abstractmethod
from typing import Generic

from hypothesis_testing.critical_values.critical_area.model import CriticalArea
from hypothesis_testing.model import CriticalValueT

from pysatl_criterion import CriticalValueCalculator, PValueCalculator


class AlternativeFactory(ABC, Generic[CriticalValueT]):
    @abstractmethod
    def get_critical_value_calculator(self) -> CriticalValueCalculator:
        pass

    @abstractmethod
    def get_p_value_calculator(self) -> PValueCalculator:
        pass

    @abstractmethod
    def get_critical_area(self, critical_value: CriticalValueT) -> CriticalArea:
        pass
