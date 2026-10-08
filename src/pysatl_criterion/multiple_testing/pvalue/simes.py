"""Simes global intersection test; does not make per-hypothesis decisions."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray, PValues, validate_alpha
from pysatl_criterion.multiple_testing.metadata import ProcedureMetadata
from pysatl_criterion.multiple_testing.result import GlobalTestResult


class Simes:
    metadata = ProcedureMetadata(
        "simes",
        "TYPE_I",
        "GLOBAL",
        "Valid intersection-null Simes inequality; independent uniform null p-values suffice. "
        "No individual null decisions or strong FWER claim.",
        ("https://doi.org/10.1093/biomet/73.3.751",),
        supports_adjusted_p_values=False,
    )

    @classmethod
    def short_code(cls) -> str:
        return cls.metadata.code

    @classmethod
    def code(cls) -> str:
        return "GLOBAL_TEST_SIMES"

    def combine(self, p_values: FloatArray) -> float:
        p = np.sort(PValues(p_values).values)
        if p.size == 0:
            return 1.0
        return min(1.0, float(np.min(p * p.size / np.arange(1, p.size + 1))))

    def test(self, data: PValues, alpha: float = 0.05) -> GlobalTestResult:
        alpha = validate_alpha(alpha)
        return GlobalTestResult(self.combine(data.values), alpha)
