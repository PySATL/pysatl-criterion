"""Shared frequency tables and validation for binning strategies."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pysatl_criterion.distribution.distribution_kind import DistributionKind
from pysatl_criterion.distribution.evaluation import BoundDistribution
from pysatl_criterion.statistics.goodness_of_fit.binning.counts import (
    FixedCount,
    positive_integer,
)


def validate_sample(rvs: ArrayLike) -> NDArray[np.float64]:
    if np.iscomplexobj(rvs) or (np.ma.isMaskedArray(rvs) and np.any(rvs.mask)):
        raise ValueError("Sample must be real and contain no masked observations")
    sample = np.asarray(rvs, dtype=np.float64)
    if sample.ndim != 1 or not sample.size or not np.all(np.isfinite(sample)):
        raise ValueError("Sample must be nonempty, one-dimensional and finite")
    return sample


@dataclass(frozen=True)
class FrequencyTable:
    observed: NDArray[np.float64]
    expected: NDArray[np.float64]

    def __post_init__(self):
        observed = np.array(self.observed, dtype=np.float64, copy=True)
        expected = np.array(self.expected, dtype=np.float64, copy=True)
        if observed.ndim != 1 or expected.shape != observed.shape or len(observed) < 2:
            raise ValueError("At least two aligned frequency categories are required")
        if not np.all(np.isfinite(observed)) or np.any(observed < 0):
            raise ValueError("Observed counts must be finite and nonnegative")
        if not np.all(np.isfinite(expected)) or np.any(expected <= 0):
            raise ValueError("Expected counts must be finite and positive")
        if not np.isclose(observed.sum(), expected.sum(), rtol=1e-10, atol=1e-10):
            raise ValueError("Observed and expected totals must match")
        observed.setflags(write=False)
        expected.setflags(write=False)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "expected", expected)


class BinningStrategy(Protocol):
    def supports(self, kind: DistributionKind) -> bool: ...

    def build(self, sample: ArrayLike, distribution: BoundDistribution) -> FrequencyTable: ...


def validate_min_expected(value):
    if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
        raise ValueError("min_expected must be positive or None")


def resolve_count(count, n, max_bins, min_expected=None):
    rule = FixedCount(count) if isinstance(count, (int, np.integer)) else count
    if not callable(getattr(rule, "resolve", None)):
        raise TypeError("count must be an integer or a CountRule")
    k = positive_integer(rule.resolve(n), "resolved count", 2)
    k = min(k, positive_integer(max_bins, "max_bins", 2))
    if min_expected is not None:
        k = min(k, int(np.floor(n / min_expected)))
    if k < 2:
        raise ValueError("Insufficient sample size for two bins with min_expected")
    return k


def make_table(observed, expected, min_expected=None):
    """Remove null-empty categories and merge adjacent small expected categories.

    Merging uses only expected frequencies, never observed discrepancies.
    Impossible observations are rejected instead of discarded.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if not np.all(np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("Invalid category probabilities")
    if np.any((expected == 0) & (observed > 0)):
        raise ValueError("Observed data in a zero-probability category")
    positive = expected > 0
    obs, exp = list(observed[positive]), list(expected[positive])
    while min_expected is not None and len(exp) > 1 and min(exp) < min_expected:
        i = int(np.argmin(exp))
        j = i + 1 if i < len(exp) - 1 else i - 1
        obs[j] += obs[i]
        exp[j] += exp[i]
        del obs[i]
        del exp[i]
    return FrequencyTable(np.asarray(obs), np.asarray(exp))


def validate_edges(edges):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 3 or np.any(np.isnan(edges)):
        raise ValueError("At least three ordered edges are required")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Edges must be strictly increasing")
    return edges


def interval_probabilities(distribution, edges):
    cdf = np.asarray(distribution.cdf(edges), dtype=float)
    sf = np.asarray(distribution.sf(edges), dtype=float)
    if cdf[0] != 0 or sf[-1] != 0:
        raise ValueError("Edges must cover the entire null distribution, including tails")
    # Survival probabilities avoid cancellation in the upper tail.
    return np.where(cdf[1:] <= 0.5, np.diff(cdf), -np.diff(sf))
