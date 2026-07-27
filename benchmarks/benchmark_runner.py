import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Store the timing result for one benchmarked implementation."""

    name: str
    total_time: float
    calls: int

    @property
    def average_time(self) -> float:
        """Return the average execution time per call."""
        return self.total_time / self.calls


class BenchmarkRunner:
    """Measure and report execution times for two implementations."""

    def __init__(self, calls: int) -> None:
        if calls <= 0:
            raise ValueError("Number of calls must be positive.")

        self.calls = calls

    def measure(self, name: str, function: Callable[[], object]) -> BenchmarkResult:
        """Measure a function for the configured number of calls."""
        start = time.perf_counter()

        for _ in range(self.calls):
            function()

        return BenchmarkResult(
            name=name,
            total_time=time.perf_counter() - start,
            calls=self.calls,
        )

    def run_comparison(
        self,
        *,
        sample_size: int,
        reference_name: str,
        reference_function: Callable[[], object],
        optimized_name: str,
        optimized_function: Callable[[], object],
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """Measure two implementations and print their comparison."""
        reference_result = self.measure(reference_name, reference_function)
        optimized_result = self.measure(optimized_name, optimized_function)

        print(f"Sample size: {sample_size}")
        print(f"Calls: {self.calls}")
        print()
        self._print_result(reference_result)
        print()
        self._print_result(optimized_result)
        print()
        print(f"Speedup: {reference_result.total_time / optimized_result.total_time:.2f}x")

        return reference_result, optimized_result

    @staticmethod
    def _print_result(result: BenchmarkResult) -> None:
        print(f"{result.name}:")
        print(f"Total time: {result.total_time:.6f} seconds")
        print(f"Average time per call: {result.average_time:.9f} seconds")
