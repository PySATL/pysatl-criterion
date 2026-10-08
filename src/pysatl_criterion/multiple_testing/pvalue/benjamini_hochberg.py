"""BenjaminiHochberg p-value adjustment; see docs/multiple_testing/benjamini_hochberg.md."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import BH, BY, ProcedureMetadata

from .base import AbstractPValueAdjustment


class BenjaminiHochberg(AbstractPValueAdjustment):
    metadata = ProcedureMetadata(
        "fdr_bh",
        "FDR",
        "STEP_UP",
        "Valid null p-values; independence or PRDS on the true null subset.",
        (BH, BY),
    )

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        return np.minimum.accumulate((p * p.size / np.arange(1, p.size + 1))[::-1])[::-1]
