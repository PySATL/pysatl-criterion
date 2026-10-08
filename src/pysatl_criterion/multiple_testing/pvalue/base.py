"""Contracts for alpha-independent and alpha-dependent p-value procedures."""

from abc import ABC, abstractmethod

import numpy as np

from pysatl_criterion.multiple_testing.base import AbstractMultipleTesting
from pysatl_criterion.multiple_testing.inputs import (
    FloatArray,
    PValues,
    validate_alpha,
    validate_p_values,
)
from pysatl_criterion.multiple_testing.result import MultipleTestingResult


class AbstractPValueAdjustment(AbstractMultipleTesting[PValues], ABC):
    """Alpha-independent adjustment. Subclasses operate on validated sorted values."""

    def adjust(self, p_values: FloatArray) -> FloatArray:
        values = validate_p_values(p_values)
        if values.size == 0:
            return values
        order = np.argsort(values, kind="stable")
        adjusted = np.clip(self._adjust_sorted(values[order]), 0.0, 1.0)
        result = np.empty_like(values)
        result[order] = adjusted
        return result

    @abstractmethod
    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        raise NotImplementedError

    def test(self, data: PValues, alpha: float = 0.05) -> MultipleTestingResult:
        alpha = validate_alpha(alpha)
        adjusted = self.adjust(data.values)
        return MultipleTestingResult(adjusted <= alpha, alpha, adjusted)


class AbstractAlphaDependentPValueProcedure(AbstractMultipleTesting[PValues], ABC):
    """No adjust(p) API: the target alpha is essential to the calculation."""
