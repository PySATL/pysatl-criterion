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

from pysatl_criterion.statistics.goodness_of_fit.hyperbolic import (
    CramerVonMisesHyperbolicGofStatistic,
)


def reference(cdf: np.ndarray) -> float:
    """Calculate the statistic using vectorized NumPy operations."""
    n = len(cdf)
    expected = (2.0 * np.arange(n) + 1.0) / (2.0 * n)
    return float(1.0 / (12.0 * n) + np.sum((expected - cdf) ** 2))


def main() -> None:
    """Compare the NumPy and Numba statistic kernels."""
    sorted_sample = create_sorted_sample()
    cdf = cdf_values(sorted_sample)
    statistic = CramerVonMisesHyperbolicGofStatistic(
        alpha=ALPHA,
        beta=BETA,
        delta=DELTA,
        mu=MU,
    )
    optimized_result = statistic.do_execute_statistic(sorted_sample, cdf)
    np.testing.assert_allclose(optimized_result, reference(cdf))

    BenchmarkRunner(CALLS).run_comparison(
        sample_size=SAMPLE_SIZE,
        reference_name="NumPy reference statistic kernel",
        reference_function=lambda: reference(cdf),
        optimized_name="Numba statistic kernel",
        optimized_function=lambda: statistic.do_execute_statistic(sorted_sample, cdf),
    )


if __name__ == "__main__":
    main()
