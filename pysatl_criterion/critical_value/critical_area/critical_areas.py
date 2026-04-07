from dataclasses import dataclass

from pysatl_criterion.critical_value.critical_area.model import CriticalArea


@dataclass
class LeftCriticalArea(CriticalArea):
    critical_value: float

    def __init__(self, critical_value: float):
        self.critical_value = critical_value

    def contains(self, value: float) -> bool:
        return value >= self.critical_value


@dataclass
class RightCriticalArea(CriticalArea):
    critical_value: float

    def __init__(self, critical_value: float):
        self.critical_value = critical_value

    def contains(self, value: float) -> bool:
        return value <= self.critical_value


@dataclass
class TwoSidedCriticalArea(CriticalArea):
    left_cv: float
    right_cv: float

    def __init__(self, left_cv: float, right_cv: float):
        self.left_cv = left_cv
        self.right_cv = right_cv

    def contains(self, value: float) -> bool:
        return self.left_cv <= value <= self.right_cv
