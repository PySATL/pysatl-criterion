# Spiegelhalter test for normality

## Description

Performs the Spiegelhalter goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on sample range, mean absolute deviation, and sample variance.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SpiegelhalterNormalityGofStatistic,
)


test_statistic = SpiegelhalterNormalityGofStatistic()
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

The null hypothesis is normality with unknown mean and variance;
`hypothesis().parameters()` is an empty dictionary. No distribution
parameters are accepted by the constructor.

The statistic combines the standardized sample range and mean absolute
deviation [1]. Location and scale cancel in the calculation. This class
uses a two-sided alternative and requires at least four observations
with nonzero sample variance.

## References

1. Spiegelhalter, D. J. (1977). "A test for normality against symmetric alternatives." Biometrika, 64(2), 415-418. [Source](https://doi.org/10.1093/biomet/64.2.415)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SpiegelhalterNormalityGofStatistic,
)


test_statistic = SpiegelhalterNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
