import numpy as np
import scipy.stats as scipy_stats
from benchmark_runner import BenchmarkRunner

from pysatl_criterion.statistics.goodness_of_fit.laplace import KuiperLaplaceGofStatistic


def kuiper_laplace_reference(sorted_sample: np.ndarray, location: float, scale: float) -> float:
    """Calculate the Kuiper statistic using the original SciPy approach."""
    n = len(sorted_sample)
    cdf_values = scipy_stats.laplace.cdf(
        sorted_sample,
        loc=location,
        scale=scale,
    )
    indices = np.arange(1, n + 1)
    d_plus = np.max(indices / n - cdf_values)
    d_minus = np.max(cdf_values - (indices - 1) / n)

    return float(d_plus + d_minus)


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

    statistic = KuiperLaplaceGofStatistic(
        t=location,
        s=scale,
    )

    optimized_result = statistic.do_execute_statistic(sorted_sample)
    reference_result = kuiper_laplace_reference(
        sorted_sample,
        location,
        scale,
    )
    np.testing.assert_allclose(optimized_result, reference_result)
    runner = BenchmarkRunner(calls)

    runner.run_comparison(
        sample_size=sample.size,
        reference_name="SciPy reference implementation",
        reference_function=lambda: kuiper_laplace_reference(
            sorted_sample,
            location,
            scale,
        ),
        optimized_name="Numba implementation",
        optimized_function=lambda: statistic.do_execute_statistic(sorted_sample),
    )


if __name__ == "__main__":
    main()
