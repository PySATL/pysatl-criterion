# Zhang Q test for normality

## Description

Performs the Zhang Q goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on linear combinations of ordered observations with normal-score weights.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangQNormalityGofStatistic,
)


test_statistic = ZhangQNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

Two linear combinations of the ordered observations use coefficients
constructed from approximate expected normal scores. The statistic is
`log(q1 / q2)`. Location cancels from each contrast and scale cancels
from their ratio. The implementation declares a right-tail alternative.

At least eight finite observations are needed because the coefficients
access the first eight normal scores. The contrasts must yield a positive,
finite ratio. Smaller samples are not supported by the current formula.

## References

1. Zhang, P. (1999). Omnibus test of normality using the Q statistic. Journal of Applied Statistics, 26(4), 519-528. [Source](https://doi.org/10.1080/02664769922395)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangQNormalityGofStatistic,
)


test_statistic = ZhangQNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
