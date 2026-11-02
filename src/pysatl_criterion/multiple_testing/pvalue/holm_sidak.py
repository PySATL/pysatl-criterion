"""HolmSidak p-value adjustment; see docs/multiple_testing/holm_sidak.md."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import HOLM, INDEPENDENT, SIDAK, ProcedureMetadata

from ._numerics import sidak_adjustment
from .base import AbstractPValueAdjustment


class HolmSidak(AbstractPValueAdjustment):
    metadata = ProcedureMetadata("holm-sidak", "FWER", "STEP_DOWN", INDEPENDENT, (HOLM, SIDAK))

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        return np.maximum.accumulate(
            sidak_adjustment(p, np.arange(p.size, 0, -1, dtype=np.float64))
        )
