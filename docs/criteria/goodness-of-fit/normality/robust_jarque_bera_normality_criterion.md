# Robust Jarque-Bera test for normality

## Description

Performs the Robust Jarque-Bera goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on robust scale, sample median, skewness, and kurtosis components.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RobustJarqueBeraNormalityGofStatistic,
)


test_statistic = RobustJarqueBeraNormalityGofStatistic()
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

The scale estimate is `sqrt(pi / 2)` times the average absolute
deviation from the sample median. Third and fourth moments are still
centered at the sample mean. The statistic combines their standardized
squares with the constants 6 and 64 used in this implementation.
Large values correspond to the right-tail alternative.

Use finite observations with positive scale. A scalar statistic is
returned; no asymptotic chi-square p-value is computed here.

## References

1. Gel, Y. R. and Gastwirth, J. L. (2008). A robust modification of the Jarque-Bera test of normality. Economics Letters, 99(1), 30-32. [Source](https://doi.org/10.1016/j.econlet.2007.05.022)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RobustJarqueBeraNormalityGofStatistic,
)


test_statistic = RobustJarqueBeraNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
