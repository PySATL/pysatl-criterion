"""Holland-Copenhaver step-down Sidak, optionally using logical null-count bounds."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray, validate_p_values
from pysatl_criterion.multiple_testing.metadata import ProcedureMetadata

from ._numerics import sidak_adjustment
from .base import AbstractPValueAdjustment


@dataclass(frozen=True)
class HollandCopenhaver(AbstractPValueAdjustment):
    """Bounds must be justified for every ordering, not estimated from observed p-values."""

    true_null_bounds: tuple[int, ...] | None = None

    metadata = ProcedureMetadata(
        "holland-copenhaver",
        "FWER",
        "STEP_DOWN",
        "Valid null p-values and independence, or the positive orthant dependence condition "
        "of Holland-Copenhaver (1987). Custom bounds must dominate the number of possible "
        "true nulls after previous rejections, for every possible ordering.",
        ("https://doi.org/10.2307/2531823",),
    )

    def __post_init__(self) -> None:
        bounds = self.true_null_bounds
        if bounds is None:
            return
        if not isinstance(bounds, tuple):
            raise TypeError("true_null_bounds must be a tuple of integers.")
        if any(type(t) is not int or not 1 <= t <= len(bounds) - i for i, t in enumerate(bounds)):
            raise ValueError("Bounds must be positive integers at most m-i at zero-based rank i.")
        if any(left < right for left, right in pairwise(bounds)):
            raise ValueError("Bounds must be nonincreasing.")

    def adjust(self, p_values: FloatArray) -> FloatArray:
        values = validate_p_values(p_values)
        if self.true_null_bounds is not None and len(self.true_null_bounds) != values.size:
            raise ValueError("One true-null bound is required per hypothesis.")
        return super().adjust(values)

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        counts = (
            np.arange(p.size, 0, -1, dtype=np.float64)
            if self.true_null_bounds is None
            else np.array(self.true_null_bounds, dtype=np.float64)
        )
        return np.maximum.accumulate(sidak_adjustment(p, counts))
