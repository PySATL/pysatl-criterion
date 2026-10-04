import numpy as np
from benchmark_runner import BenchmarkRunner
from hyperbolic_benchmark_utils import (
    ALPHA,
    BETA,
    CALLS,
    DELTA,
    MU,
    SAMPLE_SIZE,
    cdf_values,
    create_sorted_sample,
)

from pysatl_criterion.statistics.goodness_of_fit.hyperbolic import KuiperHyperbolicGofStatistic


def reference(cdf: np.ndarray) -> float:
    """Calculate the statistic using vectorized NumPy operations."""
    n = len(cdf)
    positions = np.arange(n, dtype=np.float64)
    d_plus = np.max((positions + 1.0) / n - cdf)
    d_minus = np.max(cdf - positions / n)
    return float(d_plus + d_minus)


def main() -> None:
    """Compare the NumPy and Numba statistic kernels."""
    cdf = cdf_values(create_sorted_sample())
    statistic = KuiperHyperbolicGofStatistic(
        alpha=ALPHA,
        beta=BETA,
        delta=DELTA,
        mu=MU,
    )
    optimized_result = statistic.do_execute_statistic(cdf)
    np.testing.assert_allclose(optimized_result, reference(cdf))

    BenchmarkRunner(CALLS).run_comparison(
        sample_size=SAMPLE_SIZE,
        reference_name="NumPy reference statistic kernel",
        reference_function=lambda: reference(cdf),
        optimized_name="Numba statistic kernel",
        optimized_function=lambda: statistic.do_execute_statistic(cdf),
    )


if __name__ == "__main__":
    main()
