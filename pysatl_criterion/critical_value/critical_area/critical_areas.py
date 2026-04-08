from dataclasses import dataclass

from pysatl_criterion.critical_value.critical_area.model import CriticalArea


@dataclass
class LeftCriticalArea(CriticalArea):
    """
    Critical area for left-tailed hypothesis test.
    """

    critical_value: float

    def __init__(self, critical_value: float):
        self.critical_value = critical_value

    def contains(self, value: float) -> bool:
        """
        Check if value falls within acceptance region.

        :param value: test statistic value to check.
        :return: True if value is in acceptance region, False otherwise.
        """
        return value >= self.critical_value


@dataclass
class RightCriticalArea(CriticalArea):
    """
    Critical area for right-tailed hypothesis test.
    """

    critical_value: float

    def __init__(self, critical_value: float):
        self.critical_value = critical_value

    def contains(self, value: float) -> bool:
        """
        Check if given value falls within acceptance region.

        :param value: test statistic value to check.
        :return: True if value is in acceptance region, False otherwise.
        """
        return value <= self.critical_value


@dataclass
class TwoSidedCriticalArea(CriticalArea):
    """
    Critical area for two-tailed hypothesis test.
    """

    left_cv: float
    right_cv: float

    def __init__(self, left_cv: float, right_cv: float):
        self.left_cv = left_cv
        self.right_cv = right_cv

    def contains(self, value: float) -> bool:
        """
        Check if given value falls within acceptance region.

        :param value: test statistic value to check.
        :return: True if value is in acceptance region, False otherwise.
        """
        return self.left_cv <= value <= self.right_cv
