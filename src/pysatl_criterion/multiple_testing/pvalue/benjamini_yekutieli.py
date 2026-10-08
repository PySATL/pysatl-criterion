"""BenjaminiYekutieli p-value adjustment; see docs/multiple_testing/benjamini_yekutieli.md."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import ARBITRARY, BY, ProcedureMetadata

from .base import AbstractPValueAdjustment


class BenjaminiYekutieli(AbstractPValueAdjustment):
    metadata = ProcedureMetadata("fdr_by", "FDR", "STEP_UP", ARBITRARY, (BY,))

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        harmonic = np.sum(1.0 / np.arange(1, p.size + 1))
        return np.minimum.accumulate((p * p.size * harmonic / np.arange(1, p.size + 1))[::-1])[::-1]
