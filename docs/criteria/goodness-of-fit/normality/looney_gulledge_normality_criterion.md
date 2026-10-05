# Looney-Gulledge test for normality

## Description

Performs the Looney-Gulledge goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on the correlation between ordered observations and expected normal order statistics.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LooneyGulledgeNormalityGofStatistic,
)


test_statistic = LooneyGulledgeNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The normal mean and variance are unrestricted, and
`hypothesis().parameters()` is empty. Sorted observations are
correlated with normal quantiles at `(i - 3/8)/(n + 1/4)` for
one-based ranks. This class uses the unweighted plotting positions,
including when observations are tied.

Correlation removes location and scale. Smaller coefficients indicate
departures from normality; `alternative()` returns `LeftAlternative`.
Supply finite, nonconstant observations. At least three observations
are needed for a nontrivial correlation-based normality assessment.
The returned coefficient does not include a p-value.

## References

1. Looney, S. W. and Gulledge, T. R., Jr. (1985). Use of the Correlation Coefficient with Normal Probability Plots. The American Statistician, 39(1), 75-79. [Source](https://doi.org/10.1080/00031305.1985.10479395)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LooneyGulledgeNormalityGofStatistic,
)


test_statistic = LooneyGulledgeNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
