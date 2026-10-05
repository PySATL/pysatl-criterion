# BHS test for normality

## Description

Performs the BHS goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a robust medcouple-based skewness statistic.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    BHSNormalityGofStatistic,
)


test_statistic = BHSNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The null hypothesis is normality with unknown mean and variance;
`hypothesis().parameters()` is an empty dictionary. No distribution
parameters are accepted by the constructor.

The statistic combines the medcouple with left and right medcouples in
a quadratic form [1]. These measure skewness and tail weight without
fixing the location or scale. Large values provide evidence against
normality; the class uses a right-sided alternative.

**Implementation limitation:** The current medcouple implementation is incomplete. It may fail to
terminate for samples with five or more observations and is not suitable
for reliable inference until repaired. Existing regression tests for
these sample sizes are disabled for this reason.

## References

1. Brys, G., Hubert, M., and Struyf, A. (2008). "Goodness-of-fit tests based on a robust measure of skewness." Computational Statistics, 23, 429-442. [Source](https://doi.org/10.1007/s00180-007-0083-7)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    BHSNormalityGofStatistic,
)


test_statistic = BHSNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
```
