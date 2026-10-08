"""Sidak p-value adjustment; see docs/multiple_testing/sidak.md."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import INDEPENDENT, SIDAK, ProcedureMetadata

from ._numerics import sidak_adjustment
from .base import AbstractPValueAdjustment


class Sidak(AbstractPValueAdjustment):
    metadata = ProcedureMetadata("sidak", "FWER", "SINGLE_STEP", INDEPENDENT, (SIDAK,))

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        return sidak_adjustment(p, np.full(p.size, p.size, dtype=np.float64))
