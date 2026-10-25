"""Continuous binning with explicit count and tail policies."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.statistics.goodness_of_fit.binning.base import (
    interval_probabilities,
    make_table,
    resolve_count,
    validate_edges,
    validate_min_expected,
    validate_sample,
)
from pysatl_criterion.statistics.goodness_of_fit.binning.counts import (
    CountRule,
    PowerCount,
    positive_integer,
)


class _ContinuousBins:
    def to_config(self) -> dict:
        from pysatl_criterion.statistics.goodness_of_fit.binning.configuration import (
            binning_configuration,
        )

        return binning_configuration(self)

    def supports(self, kind):
        return kind == DistributionKind.CONTINUOUS


@dataclass(frozen=True)
class EqualProbabilityBins(_ContinuousBins):
    """Equal null-probability cells; count is recomputed for each sample size.

    min_expected=None allows small samples for Monte Carlo calibration.
    Set min_expected=5 explicitly when that asymptotic policy is desired.

    References
    ----------
    Rolke, W., and Gutierrez Gongora, C. (2021). A chi-square goodness-of-fit
    test for continuous distributions against a known alternative.
    Computational Statistics, 36, 1885–1900.
    https://doi.org/10.1007/s00180-020-00997-x
    Equal-probability binning: 2020 preprint, section 5.1.
    https://arxiv.org/html/2005.02793v1#S5.SS1
    """

    count: int | CountRule = field(default_factory=PowerCount)
    min_expected: float | None = None
    max_bins: int = 1000

    def __post_init__(self):
        validate_min_expected(self.min_expected)
        positive_integer(self.max_bins, "max_bins", 2)
        resolve_count(self.count, 1, self.max_bins)

    def build(self, sample, distribution):
        sample = validate_sample(sample)
        k = resolve_count(self.count, len(sample), self.max_bins, self.min_expected)
        values = np.asarray(distribution.cdf(sample), dtype=float)
        if values.shape != sample.shape or not np.all(np.isfinite(values)):
            raise ValueError("CDF must return finite probabilities aligned with the sample")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("CDF probabilities must lie in [0, 1]")
        observed, _ = np.histogram(values, bins=np.linspace(0, 1, k + 1))
        return make_table(observed, np.full(k, len(sample) / k))


@dataclass(frozen=True)
class ExplicitBins(_ContinuousBins):
    """Explicit full-support edges; NumPy's [left, right) boundary convention.

    The final cell includes its right endpoint. Small expected cells are
    merged when min_expected is set. No observations or tails are discarded.

    References
    ----------
    Cressie, N., and Read, T. R. C. (1984). Multinomial Goodness-of-Fit Tests.
    Journal of the Royal Statistical Society, Series B, 46(3), 440–464.
    Section 1 describes counts in exhaustive, mutually exclusive categories.
    The boundary and merging policies here are implementation choices.
    https://doi.org/10.1111/j.2517-6161.1984.tb01318.x
    """

    edges: tuple[float, ...]
    min_expected: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "edges", tuple(validate_edges(self.edges)))
        validate_min_expected(self.min_expected)

    def build(self, sample, distribution):
        sample = validate_sample(sample)
        edges = np.asarray(self.edges)
        if np.any(sample < edges[0]) or np.any(sample > edges[-1]):
            raise ValueError("Sample lies outside bin edges")
        probabilities = interval_probabilities(distribution, edges)
        observed, _ = np.histogram(sample, bins=edges)
        return make_table(observed, len(sample) * probabilities, self.min_expected)


def _width_edges(distribution, lower, upper, k):
    edges = np.linspace(lower, upper, k + 1)
    if distribution.cdf(lower) > 0:
        edges = np.r_[-np.inf, edges]
    if distribution.sf(upper) > 0:
        edges = np.r_[edges, np.inf]
    return edges


@dataclass(frozen=True)
class EqualWidthBins(_ContinuousBins):
    """Equal-width central cells on fixed bounds, plus any nonempty null tails.

    count/max_bins apply to central cells; up to two tail cells are added.
    Bounds must be selected independently of the tested observations.
    The separate tail-cell policy is specific to this implementation.

    References
    ----------
    Rolke, W., and Gutierrez Gongora, C. (2021). A chi-square goodness-of-fit
    test for continuous distributions against a known alternative.
    Computational Statistics, 36, 1885–1900.
    https://doi.org/10.1007/s00180-020-00997-x
    Equal-width binning: 2020 preprint, section 5.1.
    https://arxiv.org/html/2005.02793v1#S5.SS1
    """

    lower: float
    upper: float
    count: int | CountRule = field(default_factory=PowerCount)
    min_expected: float | None = None
    max_bins: int = 1000

    def __post_init__(self):
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError("Finite bounds lower < upper are required")
        if not np.isfinite(self.upper - self.lower):
            raise ValueError("Interval width must be finite")
        validate_min_expected(self.min_expected)
        resolve_count(self.count, 1, self.max_bins)

    def build(self, sample, distribution):
        sample = validate_sample(sample)
        k = resolve_count(self.count, len(sample), self.max_bins)
        edges = _width_edges(distribution, self.lower, self.upper, k)
        return ExplicitBins(tuple(edges), self.min_expected).build(sample, distribution)


def _quantile(distribution, probability, lower, upper):
    """Invert a continuous CDF using fixed bracketing, without a backend PPF."""
    width = upper - lower
    for _ in range(100):
        if distribution.cdf(lower) <= probability <= distribution.cdf(upper):
            return brentq(lambda x: float(distribution.cdf(x)) - probability, lower, upper)
        lower -= width
        upper += width
        width *= 2
        if not np.isfinite(lower) or not np.isfinite(upper):
            break
    raise ValueError("Cannot bracket a finite null quantile; supply different bounds")


@dataclass(frozen=True)
class InterpolatedBins(EqualWidthBins):
    """Interpolate interior equal-probability and equal-width boundaries.

    kappa=0 delegates to equal probability, kappa=1 to equal width.
    For 0<kappa<1, tails stay fixed and interior boundaries are interpolated
    for the same total number of cells. Requires a continuous null CDF.
    Endpoint delegation, tail handling and numerical inversion are library
    policies; cell counts can differ at kappa=0 when tails are present.

    References
    ----------
    Rolke, W., and Gutierrez Gongora, C. (2021). A chi-square goodness-of-fit
    test for continuous distributions against a known alternative.
    Computational Statistics, 36, 1885–1900.
    https://doi.org/10.1007/s00180-020-00997-x
    Interpolation formula: 2020 preprint, section 5.1, equation (3).
    https://arxiv.org/html/2005.02793v1#S5.SS1
    """

    kappa: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.kappa) or not 0 <= self.kappa <= 1:
            raise ValueError("kappa must lie in [0, 1]")

    def build(self, sample, distribution):
        if self.kappa == 0:
            return EqualProbabilityBins(self.count, self.min_expected, self.max_bins).build(
                sample, distribution
            )
        if self.kappa == 1:
            return super().build(sample, distribution)
        sample = validate_sample(sample)
        k = resolve_count(self.count, len(sample), self.max_bins)
        edges = _width_edges(distribution, self.lower, self.upper, k)
        probabilities = np.arange(1, len(edges) - 1) / (len(edges) - 1)
        quantiles = np.array(
            [_quantile(distribution, p, self.lower, self.upper) for p in probabilities]
        )
        edges[1:-1] = (1 - self.kappa) * quantiles + self.kappa * edges[1:-1]
        return ExplicitBins(tuple(edges), self.min_expected).build(sample, distribution)
