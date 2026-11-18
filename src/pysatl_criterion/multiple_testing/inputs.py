"""Validated mathematical inputs; rows of resampling matrices are JOINT draws."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError("alpha must be a real scalar.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1).")
    return float(alpha)


def real_array(values: FloatArray, ndim: int) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu" or raw.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional real numeric array.")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("Values must be finite.")
    return result


def validate_p_values(values: FloatArray, ndim: int = 1) -> FloatArray:
    result = real_array(values, ndim)
    if np.any((result < 0) | (result > 1)):
        raise ValueError("All p-values must be in range [0,1].")
    return result


@dataclass(frozen=True)
class PValues:
    """Raw p-values, shape (m,). Empty families are allowed. Owns a read-only copy."""

    values: FloatArray

    def __post_init__(self) -> None:
        values = validate_p_values(self.values)
        values.flags.writeable = False
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ResamplingStatistics:
    """Observed (m,) and joint resampled (B, m) statistics.

    The procedure selects a common tail transformation: greater, less, or
    two-sided (absolute value). With the default greater tail, inputs must
    already be oriented so larger is more extreme. Transformations and joint
    exchangeability must be justified by the caller's experimental design.
    Shape checks cannot establish exchangeability or subset pivotality.
    """

    observed: FloatArray
    resampled: FloatArray

    def __post_init__(self) -> None:
        observed = real_array(self.observed, 1)
        resampled = real_array(self.resampled, 2)
        _validate_joint_shape(observed, resampled)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "resampled", resampled)


@dataclass(frozen=True)
class ResamplingPValues:
    """Observed raw p-values (m,) and joint resampled RAW p-values (B, m).

    Each column must use the same null calibration for observed and resampled
    values. Rows must share a permutation across hypotheses. This container
    does not compute p-values or certify that a permutation scheme is valid.
    """

    observed: FloatArray
    resampled: FloatArray

    def __post_init__(self) -> None:
        observed = validate_p_values(self.observed)
        resampled = validate_p_values(self.resampled, 2)
        _validate_joint_shape(observed, resampled)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "resampled", resampled)


def _validate_joint_shape(observed: FloatArray, resampled: FloatArray) -> None:
    if resampled.shape[0] == 0 or resampled.shape[1] != observed.size:
        raise ValueError("Joint resamples require B > 0 and shape (B, m).")
    observed.flags.writeable = False
    resampled.flags.writeable = False
