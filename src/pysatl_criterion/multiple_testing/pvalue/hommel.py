"""Hommel p-value adjustment; see docs/multiple_testing/hommel.md."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import HOMMEL, SIMES, ProcedureMetadata

from .base import AbstractPValueAdjustment


class Hommel(AbstractPValueAdjustment):
    metadata = ProcedureMetadata("hommel", "FWER", "CLOSED_TESTING", SIMES, (HOMMEL,))

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        # Closed Simes testing shortcut: each suffix is the least favorable
        # intersection of its size. Independent exhaustive closure tests verify it.
        adjusted = p.copy()
        for size in range(p.size, 1, -1):
            local = np.min(size * p[-size:] / np.arange(1, size + 1))
            adjusted[-size:] = np.maximum(adjusted[-size:], local)
            adjusted[:-size] = np.maximum(adjusted[:-size], np.minimum(size * p[:-size], local))
        return adjusted
