"""Hochberg p-value adjustment; see docs/multiple_testing/hochberg.md."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import HOCHBERG, SIMES, ProcedureMetadata

from .base import AbstractPValueAdjustment


class Hochberg(AbstractPValueAdjustment):
    metadata = ProcedureMetadata("simes-hochberg", "FWER", "STEP_UP", SIMES, (HOCHBERG,))

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        return np.minimum.accumulate((p * np.arange(p.size, 0, -1))[::-1])[::-1]
