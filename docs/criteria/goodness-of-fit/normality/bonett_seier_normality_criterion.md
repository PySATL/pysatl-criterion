# Bonett-Seier test for normality

## Description

Performs the Bonett-Seier goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a transformed measure comparing standard deviation and mean absolute deviation.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    BonettSeierNormalityGofStatistic,
)


test_statistic = BonettSeierNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The statistic compares the sample standard deviation with the average
absolute deviation about the sample mean. The ratio is log-transformed,
scaled by 13.29, centered at 3, and multiplied by `sqrt(n + 2) / 3.54`.
Departures in either direction correspond to the two-sided alternative.

Use at least four finite observations with nonzero dispersion. This class
implements the signed Geary-based component described in the article,
not its joint procedure combining two different kurtosis measures.

## References

1. Bonett, D. G. and Seier, E. (2002). A test of normality with high uniform power. Computational Statistics & Data Analysis, 40(3), 435-445. [Source](https://doi.org/10.1016/S0167-9473(02)00074-9)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    BonettSeierNormalityGofStatistic,
)


test_statistic = BonettSeierNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
