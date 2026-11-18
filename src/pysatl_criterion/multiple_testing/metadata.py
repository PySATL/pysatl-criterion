"""Discoverable descriptions; assumptions are prose, not broad guarantee flags."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProcedureMetadata:
    code: str
    error_criterion: Literal["FWER", "FDR", "TYPE_I"]
    style: Literal[
        "SINGLE_STEP", "STEP_DOWN", "STEP_UP", "CLOSED_TESTING", "ADAPTIVE", "RESAMPLING", "GLOBAL"
    ]
    assumptions: str
    references: tuple[str, ...]
    supports_adjusted_p_values: bool = True
    alpha_dependent: bool = False


DUNN = "https://doi.org/10.1080/01621459.1961.10482090"
SIDAK = "https://doi.org/10.1080/01621459.1967.10482935"
HOLM = "https://www.jstor.org/stable/4615733"
HOCHBERG = "https://doi.org/10.1093/biomet/75.4.800"
HOMMEL = "https://doi.org/10.1093/biomet/75.2.383"
BH = "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"
BY = "https://doi.org/10.1214/aos/1013699998"
BKY = "https://doi.org/10.1093/biomet/93.3.491"
ARBITRARY = "Valid (super-uniform) null p-values; arbitrary dependence."
INDEPENDENT = "Valid null p-values; mutually independent p-values."
SIMES = "Valid Simes local tests for every intersection null; independence suffices."
