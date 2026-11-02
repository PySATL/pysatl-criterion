"""Common result and safe reuse of adjusted p-values."""

from dataclasses import dataclass

import numpy as np

from .inputs import BoolArray, FloatArray, validate_alpha, validate_p_values


@dataclass(frozen=True)
class MultipleTestingResult:
    """Decisions in input order; adaptive adjusted values apply ONLY at alpha."""

    rejected: BoolArray
    alpha: float
    adjusted_p_values: FloatArray | None = None
    alpha_dependent: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_alpha(self.alpha))
        rejected = np.array(self.rejected, copy=True)
        if rejected.ndim != 1 or rejected.dtype != np.bool_:
            raise ValueError("rejected must be a one-dimensional boolean array.")
        rejected.flags.writeable = False
        object.__setattr__(self, "rejected", rejected)
        if self.adjusted_p_values is not None:
            adjusted = validate_p_values(self.adjusted_p_values)
            if adjusted.shape != rejected.shape:
                raise ValueError("Result arrays must have the same shape.")
            if not np.array_equal(rejected, adjusted <= self.alpha):
                raise ValueError("Decisions must agree with adjusted values at alpha.")
            adjusted.flags.writeable = False
            object.__setattr__(self, "adjusted_p_values", adjusted)

    def decisions_at(self, alpha: float) -> BoolArray:
        """Compare reusable values; refuse to reuse adaptive values at another alpha."""
        alpha = validate_alpha(alpha)
        if alpha == self.alpha:
            return self.rejected.copy()
        if self.alpha_dependent or self.adjusted_p_values is None:
            raise ValueError("Run the procedure again at the requested alpha.")
        return self.adjusted_p_values <= alpha


@dataclass(frozen=True)
class GlobalTestResult:
    """One global intersection-null decision, never an array of individual decisions."""

    p_value: float
    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_alpha(self.alpha))
        p = validate_p_values(np.array([self.p_value]))
        object.__setattr__(self, "p_value", float(p[0]))

    @property
    def rejected(self) -> bool:
        return self.p_value <= self.alpha
