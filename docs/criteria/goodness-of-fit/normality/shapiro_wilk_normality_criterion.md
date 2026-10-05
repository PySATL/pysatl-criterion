# Shapiro-Wilk test for normality

## Description

Performs the Shapiro-Wilk goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a squared correlation-like statistic based on ordered observations and normal order statistic weights.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ShapiroWilkNormalityGofStatistic,
)


test_statistic = ShapiroWilkNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The null leaves the mean and variance unspecified;
`hypothesis().parameters()` is empty. The statistic is a squared
weighted sum of ordered observations divided by their centered sum of
squares, and is invariant to changes of location and scale.

Order-statistic weights use a normal-quantile approximation with
polynomial adjustments, following the algorithmic approach in [2].
Smaller `W` values indicate departures from normality;
`alternative()` returns `LeftAlternative`. Supply at least three
finite observations with nonzero sample variance. No p-value is returned.

## References

1. Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3-4), 591-611. [Source](https://doi.org/10.1093/biomet/52.3-4.591)

2. Royston, P. (1995). Remark AS R94: A Remark on Algorithm AS 181: The W-test for Normality. Applied Statistics, 44(4), 547-551. [Source](https://doi.org/10.2307/2986146)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ShapiroWilkNormalityGofStatistic,
)


test_statistic = ShapiroWilkNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
