# Anderson-Darling test for normality

## Description

Performs the Anderson-Darling goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on ordered standardized observations with the log CDF and log survival function of the normal distribution.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingNormalityGofStatistic,
)


test_statistic = AndersonDarlingNormalityGofStatistic()
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

The null allows any finite mean and positive variance;
`hypothesis().parameters()` is empty. Each sample is standardized
using its mean and standard deviation with `ddof=1` before computing
the tail-weighted empirical-CDF discrepancy `A_squared`.

The returned value is the unadjusted statistic, without a finite-sample
correction or a p-value. Larger values indicate poorer fit, and
`alternative()` returns `RightAlternative`. Supply at least two
finite observations with nonzero sample variance. Calibration must
account for estimating both normal parameters from every sample.

## References

1. Anderson, T. W. and Darling, D. A. (1954). A Test of Goodness of Fit. Journal of the American Statistical Association, 49(268), 765-769. [Source](https://doi.org/10.1080/01621459.1954.10501232)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingNormalityGofStatistic,
)


test_statistic = AndersonDarlingNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
