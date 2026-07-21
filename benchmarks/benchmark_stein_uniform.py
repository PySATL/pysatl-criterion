import timeit

import numpy as np

from pysatl_criterion.statistics.goodness_of_fit.uniform import _stein_uniform_statistic


sample = np.random.default_rng(42).uniform(0.0, 1.0, 1000).astype(np.float64)

_stein_uniform_statistic(sample)

python_number = 10
numba_number = 1000

python_time = timeit.timeit(
    lambda: _stein_uniform_statistic.py_func(sample),
    number=python_number,
)

numba_time = timeit.timeit(
    lambda: _stein_uniform_statistic(sample),
    number=numba_number,
)

python_average = python_time / python_number
numba_average = numba_time / numba_number

print(f"Python: {python_average:.9f} s")
print(f"Numba:  {numba_average:.9f} s")
print(f"Speedup: {python_average / numba_average:.2f}x")
