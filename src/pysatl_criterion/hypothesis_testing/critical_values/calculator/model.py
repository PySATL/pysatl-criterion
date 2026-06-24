from abc import ABC, abstractmethod
from typing import Generic

from pysatl_criterion.hypothesis_testing.model import CriticalValueT


class CriticalValueCalculator(ABC, Generic[CriticalValueT]):
    @abstractmethod
    def calculate(
        self,
        limit_distribution: list[float],
        significance_level: float,
    ) -> CriticalValueT:
        pass
