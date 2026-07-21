import time
from collections.abc import Callable

import numpy as np

from pysatl_criterion.statistics.goodness_of_fit.uniform import SteinUniformGofStatistic


def stein_uniform_reference(sample: np.ndarray) -> float:
    """Calculate the Stein statistic using the original Python approach."""
    n = len(sample)

    if n <= 1:
        return 0.0

    total = 0.0

    for i in range(n):
        x = sample[i]

        for j in range(i + 1, n):
            y = sample[j]
            total += 0.5 * (2 * max(x, y) - 2 * x - 2 * y + x**2 + y**2)

    return float(2 * total / (n * (n - 1)))


def measure(function: Callable[[], float], calls: int) -> float:
    """Measure total execution time for the requested number of calls."""
    start = time.perf_counter()

    for _ in range(calls):
        function()

    return time.perf_counter() - start


def main() -> None:
    sample_size = 100
    calls = 1_000
    sample = np.random.default_rng(42).uniform(0.0, 1.0, sample_size)
    statistic = SteinUniformGofStatistic()

    # Compile the Numba kernel before measuring execution time.
    statistic.do_execute_statistic(sample)

    reference_elapsed = measure(
        lambda: stein_uniform_reference(sample),
        calls,
    )
    optimized_elapsed = measure(
        lambda: statistic.do_execute_statistic(sample),
        calls,
    )
    speedup = reference_elapsed / optimized_elapsed

    print(f"Sample size: {sample.size}")
    print(f"Calls: {calls}")
    print()
    print("Python reference implementation:")
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
