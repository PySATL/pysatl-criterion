"""Public computation contract, independent of the numerical backend."""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


ProbabilityResult = float | NDArray[np.float64]


class BoundDistribution(Protocol):
    """Distribution with fixed parameters and elementwise probability methods.

    Scalars (including zero-dimensional arrays) return float. Other inputs return
    float64 arrays of the same shape. NaNs propagate; infinities denote limits.
    Logarithmic tails are evaluated directly by the backend where supported.
    """

    def cdf(self, x: ArrayLike) -> ProbabilityResult:
        """Return P(X <= x)."""
        ...

    def sf(self, x: ArrayLike) -> ProbabilityResult:
        """Return P(X > x)."""
        ...

    def logcdf(self, x: ArrayLike) -> ProbabilityResult:
        """Return log P(X <= x)."""
        ...

    def logsf(self, x: ArrayLike) -> ProbabilityResult:
        """Return log P(X > x)."""
        ...
