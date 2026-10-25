# Zhang Z_K for a specified Student distribution

[All Student criteria](index.md) · [Mathematical verification](mathematical-validation.md)

Class: `ZhangKStudentGofStatistic`. Code: `ZHANG_ZK_STUDENT_GOODNESS_OF_FIT`.

## Hypothesis and formula

Fix all three parameters: `df > 0`, finite `loc`, and `scale > 0`.
Use `StudentDistributionDescriptor.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})`.
Omitted parameters remain unknown and are not supported by this criterion.
Write $u_i=F_{\nu,\mu,s}(X_{(i)})$ for the CDF values of sorted observations.

$$
Z_K=\max_i\left[r_i\log\frac{r_i}{nu_i}+(n-r_i)\log\frac{n-r_i}{n(1-u_i)}\right],\quad r_i=i-\tfrac12
$$

Half-ranks are used without an extra factor of two or normalization by n.

## Input, result and calibration

`execute_statistic(rvs, **kwargs)` accepts a nonempty, finite, one-dimensional
real sample. Ties and constants are allowed. It returns one scalar without
changing the input or retaining fitted state. Large values reject
(`RightAlternative`). Extra execution keywords are ignored.

The Student implementation evaluates log-CDF and log-survival separately.
AD, Liao–Shimokawa and Zhang kernels consume these logarithms directly;
other EDF kernels reconstruct the CDF. Unrepresentable standardization or
underflow of a required tail probability raises `ValueError`.

For a continuous fully specified null, the probability integral transform
removes the null parameters from the EDF law. Calibrate the same statistic at
the same sample size; no parameter estimation is performed. The source below
identifies the general discrepancy, not a separate Student-specific derivation.

## Example

```python
from pysatl_criterion.distribution import StudentDistributionDescriptor as Student
from pysatl_criterion.statistics.goodness_of_fit.continuous.student import (
    ZhangKStudentGofStatistic,
)

parameters = Student.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
statistic = ZhangKStudentGofStatistic(parameters)
value = statistic.execute_statistic([-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1])
print(value)
```

## Scientific source

[Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.1).
