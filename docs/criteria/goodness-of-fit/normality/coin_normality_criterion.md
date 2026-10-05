# Coin test for normality

## Description

Performs the Coin goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on normal scores and standardized ordered observations.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CoinNormalityGofStatistic,
)


test_statistic = CoinNormalityGofStatistic()
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

The ordered sample is centered and scaled using its sample mean and
standard deviation with `ddof=1`. A regression on approximate expected
normal order statistics and their cubes produces a cubic coefficient.
The returned statistic is the square of that coefficient, with large
values corresponding to the right-tail alternative.

Use at least four finite observations with nonzero sample variance.
The normal-score approximation used here reports possible inaccuracy
when the sample contains more than 2000 observations.

## References

1. Coin, D. (2008). A goodness-of-fit test for normality based on polynomial regression. Computational Statistics & Data Analysis, 52(4), 2185-2198. [Source](https://doi.org/10.1016/j.csda.2007.07.012)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CoinNormalityGofStatistic,
)


test_statistic = CoinNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
