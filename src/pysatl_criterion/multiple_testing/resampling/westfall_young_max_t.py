"""Westfall-Young maxT for a supplied, justified joint null resampling matrix."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pysatl_criterion.multiple_testing.inputs import ResamplingStatistics, validate_alpha
from pysatl_criterion.multiple_testing.metadata import ProcedureMetadata
from pysatl_criterion.multiple_testing.result import MultipleTestingResult

from .base import AbstractResamplingMultipleTesting, ResamplingConfig, extreme_adjustment


@dataclass(frozen=True)
class WestfallYoungMaxT(ResamplingConfig, AbstractResamplingMultipleTesting):
    tail: Literal["greater", "less", "two-sided"] = "greater"

    metadata = ProcedureMetadata(
        "westfall-young-max-t",
        "FWER",
        "RESAMPLING",
        "Joint exchangeable null resampling; subset pivotality for strong FWER. "
        "Comparable statistic scales; the same tail transformation on all rows. "
        "Single-step or step-down as configured.",
        (
            "https://statistics.berkeley.edu/sites/default/files/tech-reports/633.pdf",
            "https://doi.org/10.2202/1544-6115.1585",
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tail not in ("greater", "less", "two-sided"):
            raise ValueError("tail must be 'greater', 'less' or 'two-sided'.")

    def test(self, data: ResamplingStatistics, alpha: float = 0.05) -> MultipleTestingResult:
        alpha = validate_alpha(alpha)
        self.validate_exact(data.observed, data.resampled)
        observed, resampled = data.observed, data.resampled
        if self.tail == "less":
            observed, resampled = -observed, -resampled
        elif self.tail == "two-sided":
            observed, resampled = np.abs(observed), np.abs(resampled)
        adjusted = extreme_adjustment(
            observed,
            resampled,
            step_down=self.step_down,
            sampling=self.sampling,
        )
        return MultipleTestingResult(adjusted <= alpha, alpha, adjusted)
