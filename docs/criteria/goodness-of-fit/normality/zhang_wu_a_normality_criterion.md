# Zhang-Wu A test for normality

## Description

Performs the Zhang-Wu A goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on weighted logarithms of fitted normal distribution values and survival values.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangWuANormalityGofStatistic,
)


test_statistic = ZhangWuANormalityGofStatistic()
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

The calculation uses sorted normal probabilities after centering by the
sample mean and scaling by the sample standard deviation with `ddof=1`.
This implementation returns `10 * Z_A - 32`. This increasing affine
transformation preserves the right-tail rejection ordering, but critical
values must use the same transformation.

Use at least four finite observations with nonzero sample variance.
Probabilities rounded to zero or one can produce an infinite statistic.

## References

1. Zhang, J. and Wu, Y. (2005). Likelihood-ratio tests for normality. Computational Statistics & Data Analysis, 49(3), 709-721. [Source](https://doi.org/10.1016/j.csda.2004.05.034)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangWuANormalityGofStatistic,
)


test_statistic = ZhangWuANormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
