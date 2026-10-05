# Skewness test for normality

## Description

Performs the Skewness goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a transformed sample skewness statistic.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SkewNormalityGofStatistic,
)


test_statistic = SkewNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

`hypothesis().parameters()` is empty: the mean and variance are
unrestricted. The standardized third central moment is transformed to
a signed `Z` score using the sample-size-dependent approximation
in [1]. Location shifts and positive scale changes leave it unchanged.

At least eight finite observations and nonzero variance are required.
Positive and negative values correspond to opposite skewness directions.
The current `alternative()` is `RightAlternative`, so its configured
rejection direction targets large positive scores. It is not an omnibus
normality statistic; `DAPNormalityGofStatistic` combines squared
skewness and kurtosis scores. This class returns no p-value.

## References

1. D'Agostino, R. B. (1970). Transformation to normality of the null distribution of g1. Biometrika, 57(3), 679-681. [Source](https://doi.org/10.1093/biomet/57.3.679)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SkewNormalityGofStatistic,
)


test_statistic = SkewNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
