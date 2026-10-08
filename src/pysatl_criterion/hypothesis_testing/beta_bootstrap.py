"""Sample-dependent calibration for fitted Beta goodness-of-fit statistics."""

from numbers import Integral, Real

import numpy as np

from pysatl_criterion.hypothesis_testing.model import TestResult
from pysatl_criterion.statistics.goodness_of_fit.beta import (
    EbnerLiebenbergBetaGofStatistic,
    RaschkeBetaGofStatistic,
)


def parametric_bootstrap_beta(
    statistic: EbnerLiebenbergBetaGofStatistic | RaschkeBetaGofStatistic,
    rvs,
    *,
    significance_level: float = 0.05,
    n_resamples: int = 1999,
    random_state: int | np.random.Generator | None = None,
) -> TestResult:
    """Fit Beta, simulate from the fit, and refit the statistic in every replicate.

    Implements Section 2 of Ebner and Liebenberg (2021),
    https://doi.org/10.1002/sta4.341. Also calibrates the Raschke statistic,
    replacing that paper's approximate normal tables with Beta bootstrap.

    Both shapes are unknown; location and scale remain 0 and 1. Returns the
    empirical upper quantile (inverse empirical CDF), an upper-tail p-value
    with the (b+1)/(B+1) correction, and the paper's strict critical-value
    decision. The corrected p-value can give a different decision at finite B.
    Bootstrap is approximate, not an exact finite-sample test. An integer
    random_state makes repeated calls reproducible without global RNG changes.

    Invalid settings or samples raise ValueError. Fit/numerical failures
    propagate, including simulated endpoints due to floating-point rounding;
    no replicates are silently discarded or clipped. Cost is B+1 statistic
    evaluations plus one initial fit, with O(B+n) auxiliary memory.
    """
    if not isinstance(statistic, (EbnerLiebenbergBetaGofStatistic, RaschkeBetaGofStatistic)):
        raise TypeError("Expected an Ebner-Liebenberg or Raschke Beta statistic")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, Real)
        or not 0 < significance_level < 1
    ):
        raise ValueError("significance_level must be a finite number strictly between 0 and 1")
    observed = float(statistic.execute_statistic(rvs))
    sample, a, b = statistic._fit(rvs)
    rng = np.random.default_rng(random_state)
    simulated = np.empty(n_resamples)
    for i in range(n_resamples):
        simulated[i] = statistic.execute_statistic(rng.beta(a, b, size=sample.size))
    if not np.isfinite(observed) or not np.all(np.isfinite(simulated)):
        raise ValueError("Bootstrap requires finite statistics for every replicate")
    critical = float(np.quantile(simulated, 1 - significance_level, method="inverted_cdf"))
    p_value = float((1 + np.count_nonzero(simulated >= observed)) / (n_resamples + 1))
    return TestResult(
        statistic=observed,
        p_value=p_value,
        critical_value=critical,
        rejected=observed > critical,
        significance_level=float(significance_level),
    )
