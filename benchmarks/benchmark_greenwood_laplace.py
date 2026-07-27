import numpy as np
import scipy.stats as scipy_stats
from benchmark_runner import BenchmarkRunner

from pysatl_criterion.statistics.goodness_of_fit.laplace import GreenwoodLaplaceGofStatistic


def greenwood_laplace_reference(
    sorted_sample: np.ndarray,
    location: float,
    scale: float,
) -> float:
    """Calculate the Greenwood statistic using the original SciPy approach."""

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

    statistic = GreenwoodLaplaceGofStatistic(
        t=location,
        s=scale,
    )

    statistic.do_execute_statistic(sorted_sample)
    runner = BenchmarkRunner(calls)

    runner.run_comparison(
        sample_size=sample.size,
        reference_name="SciPy reference implementation",
        reference_function=lambda: greenwood_laplace_reference(
            sorted_sample,
            location,
            scale,
        ),
        optimized_name="Numba implementation",
        optimized_function=lambda: statistic.do_execute_statistic(sorted_sample),
    )


if __name__ == "__main__":
    main()
