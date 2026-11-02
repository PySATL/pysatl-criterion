"""Ordered discrete categories grouped by explicit cut points."""

from dataclasses import dataclass

import numpy as np

from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.statistics.goodness_of_fit.binning.base import (
    interval_probabilities,
    make_table,
    validate_edges,
    validate_min_expected,
    validate_sample,
)


@dataclass(frozen=True)
class DiscreteCategoryBins:
    """Cells (-inf, c1], (c1, c2], ..., (cm, inf).

    Cut points need not be integer; CDF differences preserve atom masses.
    With min_expected, adjacent low-probability categories are merged.

    References
    ----------
    Cressie, N., and Read, T. R. C. (1984). Multinomial Goodness-of-Fit Tests.
    Journal of the Royal Statistical Society, Series B, 46(3), 440–464.
    Section 1 describes counts in exhaustive, mutually exclusive categories.
    The boundary and merging policies here are implementation choices.
    https://doi.org/10.1111/j.2517-6161.1984.tb01318.x
    """

    cut_points: tuple[float, ...]
    min_expected: float | None = None

    def __post_init__(self):
        cuts = np.asarray(self.cut_points, dtype=float)
        if not np.all(np.isfinite(cuts)):
            raise ValueError("Cut points must be finite")
        validate_edges(np.r_[-np.inf, cuts, np.inf])
        object.__setattr__(self, "cut_points", tuple(cuts))
        validate_min_expected(self.min_expected)

    def to_config(self) -> dict:
        from pysatl_criterion.statistics.goodness_of_fit.binning.configuration import (
            binning_configuration,
        )

        return binning_configuration(self)

    def supports(self, kind):
        return kind == DistributionKind.DISCRETE

    def build(self, sample, distribution):
        sample = validate_sample(sample)
        edges = np.r_[-np.inf, self.cut_points, np.inf]
        probabilities = interval_probabilities(distribution, edges)
        indices = np.searchsorted(self.cut_points, sample, side="left")
        observed = np.bincount(indices, minlength=len(edges) - 1)
        return make_table(observed, len(sample) * probabilities, self.min_expected)
