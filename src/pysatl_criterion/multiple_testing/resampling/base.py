"""Separate contracts for joint statistics and joint raw p-values."""

from abc import ABC
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pysatl_criterion.multiple_testing.base import AbstractMultipleTesting
from pysatl_criterion.multiple_testing.inputs import (
    FloatArray,
    ResamplingPValues,
    ResamplingStatistics,
)


class AbstractResamplingMultipleTesting(AbstractMultipleTesting[ResamplingStatistics], ABC):
    """Extension point for joint, larger-is-more-extreme statistics (e.g. maxT)."""


class AbstractResamplingPValueTesting(AbstractMultipleTesting[ResamplingPValues], ABC):
    """Separate extension point for joint raw p-values (e.g. minP)."""


@dataclass(frozen=True)
class JointResamplingAssumptions:
    """Caller declarations, not properties inferable from a numeric matrix.

    exchangeable: observed and sampled rows come from a valid joint null
    randomization scheme. subset_pivotal: its true-null subvectors have the same
    distribution under partial and complete nulls. Both are required here for
    the claimed strong FWER guarantee.
    """

    exchangeable: bool
    subset_pivotal: bool

    def __post_init__(self) -> None:
        if self.exchangeable is not True or self.subset_pivotal is not True:
            raise ValueError("A justified exchangeable, subset-pivotal joint scheme is required.")


@dataclass(frozen=True)
class ResamplingConfig:
    assumptions: JointResamplingAssumptions
    step_down: bool = True
    sampling: Literal["monte_carlo", "exact"] = "monte_carlo"

    def __post_init__(self) -> None:
        if not isinstance(self.assumptions, JointResamplingAssumptions):
            raise TypeError("Provide explicit JointResamplingAssumptions.")
        if type(self.step_down) is not bool:
            raise TypeError("step_down must be a boolean.")
        if self.sampling not in ("monte_carlo", "exact"):
            raise ValueError("sampling must be 'monte_carlo' or 'exact'.")

    def validate_exact(self, observed: FloatArray, resampled: FloatArray) -> None:
        if self.sampling == "exact" and not np.any(np.all(resampled == observed, axis=1)):
            raise ValueError(
                "Exact enumeration must include the observed row (identity permutation)."
            )


def extreme_adjustment(
    observed: FloatArray,
    resampled: FloatArray,
    *,
    step_down: bool,
    sampling: str,
) -> FloatArray:
    """Larger-is-more-extreme counts; exact enumeration or random draws plus identity."""
    if observed.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(-observed, kind="stable")
    if step_down:
        extrema = np.maximum.accumulate(resampled[:, order][:, ::-1], axis=1)[:, ::-1]
    else:
        extrema = np.max(resampled, axis=1, keepdims=True)
    counts = np.count_nonzero(extrema >= observed[order], axis=0)
    correction = int(sampling == "monte_carlo")
    adjusted = (counts + correction) / (resampled.shape[0] + correction)
    if step_down:
        adjusted = np.maximum.accumulate(adjusted)
    result = np.empty_like(observed)
    result[order] = adjusted
    return result
