# Jarque-Bera test for normality

## Description

Performs the Jarque-Bera goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on sample skewness and kurtosis departures from their normal-distribution values.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    JBNormalityGofStatistic,
)


test_statistic = JBNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The null allows every normal mean and positive variance;
`hypothesis().parameters()` is empty. Centered, standardized moments
remove location and scale. The implementation returns
`n/6 * (skewness**2 + excess_kurtosis**2/4)` using the biased moment
estimators in `scipy.stats.skew` and `scipy.stats.kurtosis`.

Larger values indicate departures from normality;
`alternative()` returns `RightAlternative`. The input is flattened.
Supply finite observations with nonzero variance. The asymptotic
chi-squared calibration described in [1] is not a finite-sample
identity, and this class computes no p-value.

## References

1. Jarque, C. M. and Bera, A. K. (1987). A Test for Normality of Observations and Regression Residuals. International Statistical Review, 55(2), 163-172. [Source](https://doi.org/10.2307/1403192)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    JBNormalityGofStatistic,
)


test_statistic = JBNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
