# Hosking1 test for normality

## Description

Performs the Hosking1 goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on sample L-moment ratios for skewness and kurtosis.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Hosking1NormalityGofStatistic,
)


test_statistic = Hosking1NormalityGofStatistic()
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

The normal mean and variance are unrestricted;
`hypothesis().parameters()` is empty. Sample L-skewness and L-kurtosis
are ratios of linear combinations of order statistics [1]. These
ratios remove location and positive scale. Their squared deviations
from normal reference values are divided by tabulated variances.

This implementation selects fixed numerical reference constants for
`n <= 25`, `25 < n <= 50`, and `n > 50`. Supply at least four
finite, nonconstant observations. The current `alternative()` returns
`TwoSidedAlternative` although the statistic is a nonnegative
quadratic discrepancy. Reference [1] describes the L-moment foundation,
not the provenance of these implementation-specific numerical constants.

## References

1. Hosking, J. R. M. (1990). L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics. Journal of the Royal Statistical Society, Series B, 52(1), 105-124. [Source](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Hosking1NormalityGofStatistic,
)


test_statistic = Hosking1NormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
