# SWRG test for normality

## Description

Performs the SWRG goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on weights derived from normal quantiles and density values.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SWRGNormalityGofStatistic,
)


test_statistic = SWRGNormalityGofStatistic()
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

Weights are formed from second differences of normal-score density
products and normalized to unit length. The squared weighted sum of the
ordered sample is divided by its centered sum of squares. Use at least
four finite observations with nonzero sample variance.

The returned value is the raw `W_RG` ratio, not `1 - W_RG`. The
published Shapiro-Wilk-type test uses small values as evidence against
normality, while this implementation currently declares a right-tail
alternative. Account for that mismatch when using it for decisions.

## References

1. Rahman, M. M. and Govindarajulu, Z. (1997). A modification of the test of Shapiro and Wilk for normality. Journal of Applied Statistics, 24(2), 219-236. [Source](https://doi.org/10.1080/02664769723828)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SWRGNormalityGofStatistic,
)


test_statistic = SWRGNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
