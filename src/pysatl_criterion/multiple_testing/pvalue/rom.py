"""Rom step-up procedure with alpha-independent adjusted values by inversion."""

import math

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray, validate_alpha
from pysatl_criterion.multiple_testing.metadata import INDEPENDENT, ProcedureMetadata

from .base import AbstractPValueAdjustment


def _critical(size: int, alpha: float) -> list[float]:
    """c[k-1] is the threshold for k remaining hypotheses (Rom recurrence)."""
    if size == 0:
        return []
    c = [alpha]
    if size == 1:
        return c
    c.append(alpha / 2)
    power_sum = alpha
    power = alpha
    for k in range(3, size + 1):
        power *= alpha
        power_sum += power
        # log-binomial avoids overflow of binomial coefficients at large k.
        terms = [
            math.exp(
                math.lgamma(k + 1)
                - math.lgamma(j + 1)
                - math.lgamma(k - j + 1)
                + (k - j) * math.log(c[j])
            )
            for j in range(1, k - 1)
            if c[j] > 0
        ]
        c.append((power_sum - math.fsum(terms)) / k)
    if any(not math.isfinite(x) or x < 0 for x in c):
        raise ArithmeticError("Rom recurrence lost numerical accuracy for this family size.")
    return c


class Rom(AbstractPValueAdjustment):
    metadata = ProcedureMetadata(
        "rom",
        "FWER",
        "STEP_UP",
        INDEPENDENT,
        ("https://doi.org/10.1093/biomet/77.3.663",),
    )

    def critical_values(self, size: int, alpha: float = 0.05) -> FloatArray:
        """Rank-ordered rejection thresholds, NOT adjusted p-values."""
        alpha = validate_alpha(alpha)
        if type(size) is not int or size < 0:
            raise ValueError("size must be a nonnegative integer.")
        return np.array(_critical(size, alpha)[::-1], dtype=np.float64)

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        inverted = np.empty_like(p)
        for i, value in enumerate(p):
            size = p.size - i
            if value == 0 or size == 1:
                inverted[i] = value
                continue
            low, high = 0.0, min(1.0, size * float(value))
            if _critical(size, high)[-1] < value:
                inverted[i] = 1.0
                continue
            # Invert the critical-value function, rather than treating an
            # alpha-specific multiplier as an alpha-independent adjustment.
            for _ in range(60):
                middle = (low + high) / 2
                if _critical(size, middle)[-1] >= value:
                    high = middle
                else:
                    low = middle
            inverted[i] = high
        return np.minimum.accumulate(inverted[::-1])[::-1]
