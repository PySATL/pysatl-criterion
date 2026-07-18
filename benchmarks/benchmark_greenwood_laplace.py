import time
from collections.abc import Callable

import numpy as np
import scipy.stats as scipy_stats

from pysatl_criterion.statistics.goodness_of_fit.laplace import GreenwoodLaplaceGofStatistic


def greenwood_laplace_reference(
    sample: np.ndarray,
    location: float,
    scale: float,
) -> float:
    """Calculate the Greenwood statistic using the original SciPy approach."""

    sorted_sample = np.sort(sample)
    cdf_values = scipy_stats.laplace.cdf(
        sorted_sample,
        loc=location,
        scale=scale,
    )
    spacings = np.diff(
        np.concatenate(
            (
                [0.0],
                cdf_values,
                [1.0],
            )
        )
    )

    return float(np.sum(spacings**2))


def measure(
    function: Callable[[], float],
    calls: int,
) -> float:
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

    statistic = GreenwoodLaplaceGofStatistic(
        t=location,
        s=scale,
    )

    # Compile the Numba function before measuring execution time.
    statistic.execute_statistic(sample)

    reference_elapsed = measure(
        lambda: greenwood_laplace_reference(
            sample,
            location,
            scale,
        ),
        calls,
    )

    optimized_elapsed = measure(
        lambda: statistic.execute_statistic(sample),
        calls,
    )

    speedup = reference_elapsed / optimized_elapsed

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
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
