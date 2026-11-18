"""Bonferroni p-value adjustment; see docs/multiple_testing/bonferroni.md."""

from pysatl_criterion.multiple_testing.inputs import FloatArray
from pysatl_criterion.multiple_testing.metadata import ARBITRARY, DUNN, ProcedureMetadata

from .base import AbstractPValueAdjustment


class Bonferroni(AbstractPValueAdjustment):
    metadata = ProcedureMetadata("bonferroni", "FWER", "SINGLE_STEP", ARBITRARY, (DUNN,))

    def _adjust_sorted(self, p: FloatArray) -> FloatArray:
        return p * p.size
