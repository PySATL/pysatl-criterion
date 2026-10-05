# D'Agostino-Pearson test for normality

## Description

Performs the D'Agostino-Pearson goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on the sum of squared transformed skewness and kurtosis statistics.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DAPNormalityGofStatistic,
)


test_statistic = DAPNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The null is the normal location-scale family, so
`hypothesis().parameters()` is empty. The implementation adds the
squared transformed skewness and kurtosis scores returned by
`SkewNormalityGofStatistic` and `KurtosisNormalityGofStatistic`.
Centered standardized moments eliminate the unknown mean and variance.

At least eight finite observations with nonzero variance are required.
Larger scores indicate departures from normality;
`alternative()` returns `RightAlternative`. The chi-squared
approximation is asymptotic; this class returns a statistic only.

## References

1. D'Agostino, R. B. and Pearson, E. S. (1973). Tests for departure from normality. Empirical results for the distributions of b2 and sqrt(b1). Biometrika, 60(3), 613-622. [Source](https://doi.org/10.1093/biomet/60.3.613)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DAPNormalityGofStatistic,
)


test_statistic = DAPNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
