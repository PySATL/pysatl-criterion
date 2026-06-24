from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pysatl_criterion.statistics import AbstractGoodnessOfFitStatistic


CriticalValue = float | tuple[float, float]
CriticalValueT = TypeVar("CriticalValueT", float, tuple[float, float])


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float | None
    critical_value: CriticalValue | None
    rejected: bool
    significance_level: float


class DecisionMethod(ABC):
    @abstractmethod
    def decide(
        self,
        statistic: AbstractGoodnessOfFitStatistic,
        statistic_value: float,
        significance_level: float,
        sample_size: int,
    ) -> TestResult:
        pass
