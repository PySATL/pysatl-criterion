# Ratio test for beta distribution

## Description

No scientific source for this exact Beta formula was identified. This class is
a locally defined discrepancy, not a claimed named published criterion.

Performs ratio goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares the sample ratio of geometric mean to arithmetic mean with the theoretical beta distribution ratio.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RatioBetaGofStatistic,
)


test_statistic = RatioBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
The sample statistic is the ratio of the geometric mean to the arithmetic mean.
The theoretical ratio uses

$$ E[X] = \frac{\alpha}{\alpha + \beta} $$

and

$$ E[\log X] = \psi(\alpha) - \psi(\alpha + \beta), $$

where $\psi$ is the digamma function.
The reference ratio `exp(E[log X]) / E[X]` is a probability limit, not the exact
finite-sample expectation. The statistic is `sqrt(n) * abs(sample_ratio - reference_ratio)`;
its variance is not standardized. Calibrate by simulation for both fixed shapes
and sample size. Zero observations give geometric mean zero; an all-zero sample
raises `ValueError` because its arithmetic mean is zero.

This statistic compares selected distribution functionals and need not detect every
non-Beta alternative. Previously generated critical values must be recalculated
when the statistic or its estimator settings change.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References

No scientific article describing the exact implemented Beta statistic was identified.
The formula is a locally defined discrepancy; no reference to a different test is substituted.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RatioBetaGofStatistic,
)


test_statistic = RatioBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
