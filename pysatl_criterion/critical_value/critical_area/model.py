from abc import ABC, abstractmethod


class CriticalArea(ABC):
    """
    Abstract base class for defining critical areas in hypothesis testing.
    """

    @abstractmethod
    def contains(self, value: float) -> bool:
        """
        Check if value falls within acceptance region.

        :param value: test statistic value to check.
        :return: True if value is in acceptance region, False otherwise.
        """
        pass
