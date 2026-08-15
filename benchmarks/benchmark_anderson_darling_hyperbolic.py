import numpy as np
from benchmark_runner import BenchmarkRunner
from hyperbolic_benchmark_utils import (
    ALPHA,
    BETA,
    CALLS,
    DELTA,
    MU,
    SAMPLE_SIZE,
    create_sorted_sample,
    log_probabilities,
)

from pysatl_criterion.statistics.goodness_of_fit.hyperbolic import (
    AndersonDarlingHyperbolicGofStatistic,
)


def reference(log_cdf: np.ndarray, log_sf: np.ndarray) -> float:
    """Calculate the statistic using vectorized NumPy operations."""
    n = len(log_cdf)
    weights = (2.0 * np.arange(n) + 1.0) / n
    return float(-n - np.sum(weights * (log_cdf + log_sf[::-1])))


def main() -> None:
    """Compare the NumPy and Numba statistic kernels."""
    sorted_sample = create_sorted_sample()
    log_cdf, log_sf = log_probabilities(sorted_sample)
    statistic = AndersonDarlingHyperbolicGofStatistic(
        alpha=ALPHA,
        beta=BETA,
        delta=DELTA,
        mu=MU,
    )
    optimized_result = statistic.do_execute_statistic(
        sorted_sample,
        log_cdf=log_cdf,
        log_sf=log_sf,
    )
    np.testing.assert_allclose(optimized_result, reference(log_cdf, log_sf))

    BenchmarkRunner(CALLS).run_comparison(
        sample_size=SAMPLE_SIZE,
        reference_name="NumPy reference statistic kernel",
        reference_function=lambda: reference(log_cdf, log_sf),
        optimized_name="Numba statistic kernel",
        optimized_function=lambda: statistic.do_execute_statistic(
            sorted_sample,
            log_cdf=log_cdf,
            log_sf=log_sf,
        ),
    )


if __name__ == "__main__":
    main()
