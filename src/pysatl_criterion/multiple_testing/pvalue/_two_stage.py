"""Exactly two stages, matching the documented maxiter=1 convention."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import PValues
from pysatl_criterion.multiple_testing.result import MultipleTestingResult

from .benjamini_hochberg import BenjaminiHochberg


def two_stage(data: PValues, alpha: float, factor: float) -> MultipleTestingResult:
    bh = BenjaminiHochberg().adjust(data.values)
    m = bh.size
    first_rejections = int(np.count_nonzero(bh <= alpha / factor))
    # Stop at zero/all rejections, avoiding division by zero and matching the
    # documented first-stage adjusted values in these two cases.
    fraction = (m - first_rejections) / m if 0 < first_rejections < m else 1.0
    adjusted = np.clip(bh * fraction * factor, 0.0, 1.0)
    return MultipleTestingResult(adjusted <= alpha, alpha, adjusted, alpha_dependent=True)
