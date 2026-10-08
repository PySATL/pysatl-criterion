"""Reproducible finite-sample calibration of zero-origin exponential statistics."""

from numbers import Integral, Real

import numpy as np

from pysatl_criterion.hypothesis_testing.model import TestResult
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.exponent import AbstractExponentialityGofStatistic


def monte_carlo_exponential(
    statistic: AbstractExponentialityGofStatistic,
    rvs,
    *,
    significance_level: float = 0.05,
    n_resamples: int = 1999,
    random_state: int | np.random.Generator | None = None,
) -> TestResult:
    """Simulate the actual statistic, including refitting, with a local Generator.

    Composite scale-invariant statistics simulate Exp(rate=1); a specified
    hypothesis simulates Exp(rate=lam). Settings are held fixed in all replicates.
    Right/left p=(1+number at least as extreme)/(B+1). Two-tail p is twice the
    smaller corrected tail probability, capped at one. Ties are conservative.
    Rejection uses p<=alpha; critical_value is an empirical quantile summary,
    which need not reproduce that decision exactly for finite B or tied data.
    This is a Monte Carlo test, not an asymptotic p-value or a fitted KS table.
    Its validity assumes the declared family and scale invariance are correct;
    calibration alone cannot validate an unpublished statistic's provenance.
    """
    if not isinstance(statistic, AbstractExponentialityGofStatistic):
        raise TypeError("Expected an exponential goodness-of-fit statistic")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, Real)
        or not 0 < significance_level < 1
    ):
        raise ValueError("significance_level must be between 0 and 1")
    observed = float(statistic.execute_statistic(rvs))
    if np.isnan(observed):
        raise ValueError("Observed statistic is NaN")
    rate = statistic.hypothesis().parameters().get("lam", 1.0)
    scale = 1.0 / rate
    if not np.isfinite(scale):
        raise ValueError("The specified exponential scale is not representable")
    rng = np.random.default_rng(random_state)
    simulated = np.array(
        [statistic.execute_statistic(rng.exponential(scale, len(rvs))) for _ in range(n_resamples)]
    )
    if not np.all(np.isfinite(simulated)):
        raise ValueError("Nonfinite simulated statistic; no replicates discarded")
    upper = (1 + np.count_nonzero(simulated >= observed)) / (n_resamples + 1)
    lower = (1 + np.count_nonzero(simulated <= observed)) / (n_resamples + 1)
    tail = statistic.alternative().type()
    critical: float | tuple[float, float]
    if tail == AlternativeType.TWO_TAILED:
        p_value = min(1.0, 2 * min(upper, lower))
        limits = np.quantile(simulated, [significance_level / 2, 1 - significance_level / 2])
        critical = (float(limits[0]), float(limits[1]))
    else:
        p_value = upper if tail == AlternativeType.RIGHT else lower
        q = 1 - significance_level if tail == AlternativeType.RIGHT else significance_level
        critical = float(np.quantile(simulated, q))
    return TestResult(
        observed, float(p_value), critical, p_value <= significance_level, float(significance_level)
    )
