# Corrected Kuiper for a specified Student distribution

[All Student criteria](index.md) · [Mathematical verification](mathematical-validation.md)

Class: `KuiperCorrectedStudentGofStatistic`. Code: `KUIPER_CORRECTED_STUDENT_GOODNESS_OF_FIT`.

## Hypothesis and formula

Fix all three parameters: `df > 0`, finite `loc`, and `scale > 0`.
Use `StudentDistributionDescriptor.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})`.
Omitted parameters remain unknown and are not supported by this criterion.
Write $u_i=F_{\nu,\mu,s}(X_{(i)})$ for the CDF values of sorted observations.

$$
V_c=\left(\sqrt n+0.155+\frac{0.24}{\sqrt n}\right)(D^++D^-)
$$

Here $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$. The conventional multiplier is part of the statistic; it does not itself compute a p-value.

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
    KuiperCorrectedStudentGofStatistic,
)

parameters = Student.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1})
statistic = KuiperCorrectedStudentGofStatistic(parameters)
value = statistic.execute_statistic([-3.2, -1.1, -0.3, 0.2, 0.7, 1.8, 4.1])
print(value)
```

## Scientific source

[Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X), equations (1)–(3).
