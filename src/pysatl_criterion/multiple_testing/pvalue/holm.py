"""Holm p-value adjustment; see docs/multiple_testing/holm.md."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import ARBITRARY, HOLM, ProcedureMetadata

from .base import AbstractPValueAdjustment


class Holm(AbstractPValueAdjustment):
    metadata = ProcedureMetadata("holm", "FWER", "STEP_DOWN", ARBITRARY, (HOLM,))

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        return np.maximum.accumulate(p * np.arange(p.size, 0, -1))
