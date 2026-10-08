"""Westfall-Young minP for jointly calibrated raw p-values (not test statistics)."""

from dataclasses import dataclass

from pysatl_criterion.multiple_testing.inputs import ResamplingPValues, validate_alpha
from pysatl_criterion.multiple_testing.metadata import ProcedureMetadata
from pysatl_criterion.multiple_testing.result import MultipleTestingResult

from .base import AbstractResamplingPValueTesting, ResamplingConfig, extreme_adjustment


@dataclass(frozen=True)
class WestfallYoungMinP(ResamplingConfig, AbstractResamplingPValueTesting):
    metadata = ProcedureMetadata(
        "westfall-young-min-p",
        "FWER",
        "RESAMPLING",
        "Joint exchangeable null resampling; subset pivotality for strong FWER. "
        "Observed and resampled raw p-values use the same marginal null calibration. "
        "Single-step or step-down as configured.",
        (
            "https://statistics.berkeley.edu/sites/default/files/tech-reports/633.pdf",
            "https://doi.org/10.2202/1544-6115.1585",
        ),
    )

    def test(self, data: ResamplingPValues, alpha: float = 0.05) -> MultipleTestingResult:
        alpha = validate_alpha(alpha)
        self.validate_exact(data.observed, data.resampled)
        adjusted = extreme_adjustment(
            -data.observed,
            -data.resampled,
            step_down=self.step_down,
            sampling=self.sampling,
        )
        return MultipleTestingResult(adjusted <= alpha, alpha, adjusted)
