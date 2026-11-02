"""BenjaminiKriegerYekutieli; alpha-dependent two-stage procedure."""

from pysatl_criterion.multiple_testing.inputs import PValues, validate_alpha
from pysatl_criterion.multiple_testing.metadata import BKY, ProcedureMetadata
from pysatl_criterion.multiple_testing.result import MultipleTestingResult

from ._two_stage import two_stage
from .base import AbstractAlphaDependentPValueProcedure


class BenjaminiKriegerYekutieli(AbstractAlphaDependentPValueProcedure):
    """Two-stage BKY, Definition 6 / Theorem 1 of the authors' manuscript."""

    metadata = ProcedureMetadata(
        "fdr_tsbky",
        "FDR",
        "ADAPTIVE",
        "Valid null p-values and mutually independent test statistics; "
        "no general PRDS guarantee is asserted for this adaptive method.",
        (BKY,),
        alpha_dependent=True,
    )

    def test(self, data: PValues, alpha: float = 0.05) -> MultipleTestingResult:
        alpha = validate_alpha(alpha)
        return two_stage(data, alpha, 1.0 + alpha)
