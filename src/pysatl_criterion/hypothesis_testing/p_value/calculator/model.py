from abc import ABC, abstractmethod


class PValueCalculator(ABC):
    @abstractmethod
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:
        pass
