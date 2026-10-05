# Doornik-Hansen test for normality

## Description

Performs the Doornik-Hansen goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on transformed skewness and kurtosis components.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DoornikHansenNormalityGofStatistic,
)


test_statistic = DoornikHansenNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The statistic is the sum of squares of transformed sample skewness and
kurtosis. Both moments are computed about the sample mean, so neither a
population mean nor a population variance is supplied. Large values
correspond to the right-tail alternative.

The formulas require a nonconstant sample and can be undefined at very
small sample sizes. The reference studies the small-sample approximation
from ten observations. This class computes the statistic only; it does not
apply the paper's asymptotic chi-square calibration.

## References

1. Doornik, J. A. and Hansen, H. (2008). An Omnibus Test for Univariate and Multivariate Normality. Oxford Bulletin of Economics and Statistics, 70(s1), 927-939. [Source](https://doi.org/10.1111/j.1468-0084.2008.00537.x)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DoornikHansenNormalityGofStatistic,
)


test_statistic = DoornikHansenNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
