# SF test for normality

## Description

Performs the SF goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a Shapiro-Francia-style statistic based on expected normal scores.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SFNormalityGofStatistic,
)


test_statistic = SFNormalityGofStatistic()
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

The null leaves mean and variance unrestricted;
`hypothesis().parameters()` is empty. The statistic is the squared
correlation between sorted observations and approximate expected
normal order statistics, using `(i - 3/8)/(n + 1/4)` plotting
probabilities. This quantile approximation is discussed in [2].

Location and scale cancel from the ratio. Smaller values indicate
nonnormality; `alternative()` returns `LeftAlternative`. Supply
finite, nonconstant data and at least three observations for a
nontrivial assessment. No p-value is calculated.

## References

1. Shapiro, S. S. and Francia, R. S. (1972). An Approximate Analysis of Variance Test for Normality. Journal of the American Statistical Association, 67(337), 215-216. [Source](https://doi.org/10.1080/01621459.1972.10481232)

2. Weisberg, S. and Bingham, C. (1975). An Approximate Analysis of Variance Test for Non-Normality Suitable for Machine Calculation. Technometrics, 17(1), 133-134. [Source](https://doi.org/10.1080/00401706.1975.10489283)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SFNormalityGofStatistic,
)


test_statistic = SFNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
