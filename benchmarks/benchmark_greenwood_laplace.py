import time

import numpy as np

from pysatl_criterion.statistics.goodness_of_fit.laplace import GreenwoodLaplaceGofStatistic


# Baseline without Numba:
# 1000 calls, sample size 10000
# Total time: 0.293953 seconds
# Average time per call: 0.000293953 seconds
#
# Optimized with Numba:
# 1000 calls, sample size 10000
# Total time: 0.199217 seconds
# Average time per call: 0.000199217 seconds


def main() -> None:
    sample = np.random.default_rng(42).laplace(
        loc=0.0,
        scale=1.0,
        size=10_000,
    )

    statistic = GreenwoodLaplaceGofStatistic(
        t=0.0,
        s=1.0,
    )

    calls = 1_000

    statistic.execute_statistic(sample)  # Warm-up call to avoid JIT compilation overhead

    start = time.perf_counter()

    for _ in range(calls):
        statistic.execute_statistic(sample)

    elapsed = time.perf_counter() - start

    print(f"Sample size: {sample.size}")
    print(f"Calls: {calls}")
    print(f"Total time: {elapsed:.6f} seconds")
    print(f"Average time per call: {elapsed / calls:.9f} seconds")


if __name__ == "__main__":
    main()
