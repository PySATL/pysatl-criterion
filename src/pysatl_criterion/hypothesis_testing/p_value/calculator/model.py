from abc import ABC, abstractmethod


class PValueCalculator(ABC):
    """
    Abstract calculator for p-values of hypothesis tests.
    """

    @abstractmethod
    def calculate(
        self,
        limit_distribution: list[float],
        statistic_value: float,
    ) -> float:
        """
        Calculate a p-value from a limit distribution and statistic value.

        :param limit_distribution: simulated or theoretical limit distribution values.
        :param statistic_value: computed statistic value for the observed sample.
        :return: p-value for the calculator type.
        """
        pass
