# Bontemps-Meddahi1 test for normality

## Description

Performs the Bontemps-Meddahi1 goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on Hermite-polynomial components of standardized observations up to orders three and four.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    BontempsMeddahi1NormalityGofStatistic,
)


test_statistic = BontempsMeddahi1NormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The sample is centered and scaled using its mean and standard deviation
with `ddof=1`. Sums of the normalized third and fourth Hermite
polynomials are squared and divided by the sample size. Large values
correspond to the right-tail alternative.

Use at least four finite observations with nonzero sample variance.
This is the independent-observation version; the covariance correction
for dependent time series discussed in the article is not implemented.

## References

1. Bontemps, C. and Meddahi, N. (2005). Testing normality: a GMM approach. Journal of Econometrics, 124(1), 149-186. [Source](https://doi.org/10.1016/j.jeconom.2004.02.014)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    BontempsMeddahi1NormalityGofStatistic,
)


test_statistic = BontempsMeddahi1NormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
