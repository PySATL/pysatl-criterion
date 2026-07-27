import time
from collections.abc import Callable

import numpy as np
import scipy.stats as scipy_stats

from pysatl_criterion.statistics.goodness_of_fit.laplace import AndersonDarlingLaplaceGofStatistic


def anderson_darling_laplace_reference(
    sorted_sample: np.ndarray, location: float, scale: float
) -> float:
    """Calculate the Anderson--Darling statistic using the original approach."""
    n = len(sorted_sample)
    log_cdf = scipy_stats.laplace.logcdf(
        sorted_sample,
        loc=location,
        scale=scale,
    )
    log_sf = scipy_stats.laplace.logsf(
        sorted_sample,
        loc=location,
        scale=scale,
    )
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1.0) / n * (log_cdf + log_sf[::-1])))


def measure(function: Callable[[], float], calls: int) -> float:
    """Measure total execution time for the requested number of calls."""
    start = time.perf_counter()

    for _ in range(calls):
        function()

    return time.perf_counter() - start


def main() -> None:
    location = 0.0
    scale = 1.0
    calls = 1_000
    sample = np.random.default_rng(42).laplace(
        loc=location,
        scale=scale,
        size=10_000,
    )
    sorted_sample = np.sort(sample)
    statistic = AndersonDarlingLaplaceGofStatistic(
        t=location,
        s=scale,
    )

    # Compile the Numba kernel before measuring execution time.
    optimized_result = statistic.do_execute_statistic(sorted_sample)
    reference_result = anderson_darling_laplace_reference(
        sorted_sample,
        location,
        scale,
    )
    np.testing.assert_allclose(optimized_result, reference_result)

    reference_elapsed = measure(
        lambda: anderson_darling_laplace_reference(
            sorted_sample,
            location,
            scale,
        ),
        calls,
    )
    optimized_elapsed = measure(
        lambda: statistic.do_execute_statistic(sorted_sample),
        calls,
    )

    print(f"Sample size: {sample.size}")
    print(f"Calls: {calls}")
    print()
    print("SciPy reference implementation:")
    print(f"Total time: {reference_elapsed:.6f} seconds")
    print(f"Average time per call: {reference_elapsed / calls:.9f} seconds")
    print()
    print("Numba implementation:")
    print(f"Total time: {optimized_elapsed:.6f} seconds")
    print(f"Average time per call: {optimized_elapsed / calls:.9f} seconds")
    print()
    print(f"Speedup: {reference_elapsed / optimized_elapsed:.2f}x")


if __name__ == "__main__":
    main()
