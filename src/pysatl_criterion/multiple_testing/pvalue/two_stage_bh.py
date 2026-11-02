"""TwoStageBenjaminiHochberg; alpha-dependent two-stage procedure."""

from pysatl_criterion.multiple_testing.inputs import PValues, validate_alpha
from pysatl_criterion.multiple_testing.metadata import BKY, ProcedureMetadata
from pysatl_criterion.multiple_testing.result import MultipleTestingResult

from ._two_stage import two_stage
from .base import AbstractAlphaDependentPValueProcedure


class TwoStageBenjaminiHochberg(AbstractAlphaDependentPValueProcedure):
    """Uncorrected two-stage BH variant; nominal FDR control is not asserted."""

    metadata = ProcedureMetadata(
        "fdr_tsbh",
        "FDR",
        "ADAPTIVE",
        "Exploratory uncorrected two-stage BH; nominal FDR control is not guaranteed, "
        "even under independence. Prefer two-stage BKY for proven control.",
        (BKY,),
        alpha_dependent=True,
    )

    def test(self, data: PValues, alpha: float = 0.05) -> MultipleTestingResult:
        alpha = validate_alpha(alpha)
        return two_stage(data, alpha, 1.0)
