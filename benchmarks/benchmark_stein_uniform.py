import numpy as np
from benchmark_runner import BenchmarkRunner

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


def main() -> None:
    sample_size = 100
    calls = 1_000
    sample = np.random.default_rng(42).uniform(0.0, 1.0, sample_size)
    statistic = SteinUniformGofStatistic()

    # Compile the Numba kernel before measuring execution time.
    statistic.do_execute_statistic(sample)
    runner = BenchmarkRunner(calls)

    runner.run_comparison(
        sample_size=sample.size,
        reference_name="Python reference implementation",
        reference_function=lambda: stein_uniform_reference(sample),
        optimized_name="Numba implementation",
        optimized_function=lambda: statistic.do_execute_statistic(sample),
    )


if __name__ == "__main__":
    main()
