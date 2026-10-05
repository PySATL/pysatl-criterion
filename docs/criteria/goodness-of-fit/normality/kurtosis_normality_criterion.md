# Kurtosis test for normality

## Description

Performs the Kurtosis goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a transformed sample kurtosis statistic.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KurtosisNormalityGofStatistic,
)


test_statistic = KurtosisNormalityGofStatistic()
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
`hypothesis().parameters()` is empty. The implementation transforms
the standardized fourth central moment to a signed `Z` score using
the finite-sample moment approximations in [1].

At least five finite observations with nonzero variance are required;
the normal approximation can be unreliable for small samples. Negative
and positive scores describe opposite kurtosis departures. The current
`alternative()` returns `RightAlternative` and therefore targets
large positive scores. Use `DAPNormalityGofStatistic` for the joint
squared skewness and kurtosis statistic. No p-value is returned.

## References

1. Anscombe, F. J. and Glynn, W. J. (1983). Distribution of the kurtosis statistic b2 for normal samples. Biometrika, 70(1), 227-234. [Source](https://doi.org/10.1093/biomet/70.1.227)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KurtosisNormalityGofStatistic,
)


test_statistic = KurtosisNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
