import numpy as np
import scipy.stats as scipy_stats
from benchmark_runner import BenchmarkRunner

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
    runner = BenchmarkRunner(calls)

    runner.run_comparison(
        sample_size=sample.size,
        reference_name="SciPy reference implementation",
        reference_function=lambda: anderson_darling_laplace_reference(
            sorted_sample,
            location,
            scale,
        ),
        optimized_name="Numba implementation",
        optimized_function=lambda: statistic.do_execute_statistic(sorted_sample),
    )


if __name__ == "__main__":
    main()
