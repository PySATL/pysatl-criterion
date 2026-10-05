# Lilliefors test for normality

## Description

Performs the Lilliefors goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on the Kolmogorov-Smirnov type discrepancy after standardizing the sample by its empirical mean and standard deviation.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsNormalityGofStatistic,
)


test_statistic = LillieforsNormalityGofStatistic()
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

The null is the entire normal location-scale family;
`hypothesis().parameters()` is empty. The mean and standard deviation
with `ddof=1` are estimated from each sample. The maximum absolute
empirical-CDF discrepancy is then evaluated on standardized observations.

The statistic is invariant to location and scale. It requires a
nonconstant finite sample of at least two observations. Its null
distribution differs from that of KS with specified parameters [1].
Larger discrepancies indicate poorer fit. The current shared Lilliefors
interface reports `TwoSidedAlternative`; this method returns only
the discrepancy, without a p-value.

## References

1. Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for Normality with Mean and Variance Unknown. Journal of the American Statistical Association, 62(318), 399-402. [Source](https://doi.org/10.1080/01621459.1967.10482916)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsNormalityGofStatistic,
)


test_statistic = LillieforsNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
