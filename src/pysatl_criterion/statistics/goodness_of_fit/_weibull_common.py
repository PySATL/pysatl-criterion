"""Shared numerical kernels for ordinary and exponentiated Weibull statistics."""

import numpy as np


def _sample(rvs, *, min_size=1, positive=False, nonconstant=False):
    if np.iscomplexobj(rvs):
        raise ValueError("Sample must be real")
    x = np.asarray(rvs, dtype=float)
    if x.ndim != 1 or x.size < min_size:
        raise ValueError(f"Sample must be one-dimensional with at least {min_size} observations")
    if not np.all(np.isfinite(x)) or np.any(x < 0) or (positive and np.any(x == 0)):
        raise ValueError(
            "Expected a finite positive sample"
            if positive
            else "Sample must contain finite nonnegative observations"
        )
    if nonconstant and np.ptp(x) == 0:
        raise ValueError("A nonconstant sample is required")
    return x


def _log_probabilities(z, a=1):
    """Log probabilities of (1-exp(-exp(z)))**a, retaining both tails."""
    z = np.asarray(z)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        t = np.exp(z)
        log_cdf = np.empty_like(z)
        small = z < -36
        large = t > 36
        middle = ~(small | large)
        log_cdf[small] = a * z[small]
        log_cdf[middle] = a * np.log(-np.expm1(-t[middle]))
        log_cdf[large] = -np.exp(np.log(a) - t[large])
        log_sf = np.log(-np.expm1(log_cdf))
        upper_tail = large & (np.log(a) - t < -36)
        log_sf[upper_tail] = np.log(a) - t[upper_tail]
    return log_cdf, log_sf


def _weighted_edf(log_cdf, log_sf):
    u = np.exp(log_cdf)
    n = len(u)
    i = np.arange(1, n + 1)
    d = np.maximum(i / n - u, u - (i - 1) / n)
    with np.errstate(over="ignore"):
        return float(np.sum(np.exp(np.log(d) - (log_cdf + log_sf) / 2)) / np.sqrt(n))
