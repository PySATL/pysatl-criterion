# GMG test for normality

## Description

Performs the GMG goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on the ratio of sample standard deviation to a robust median-based scale estimate.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GMGNormalityGofStatistic,
)


test_statistic = GMGNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The statistic divides the sample standard deviation with `ddof=0` by
`sqrt(pi / 2)` times the average absolute deviation from the sample
median. This ratio is close to one under normality, and large values
correspond to the right-tail alternative aimed at heavy-tailed departures.

Use at least four finite observations with positive dispersion.
The returned value is the raw ratio; no large-sample centering, scaling,
or p-value approximation from the article is applied.

## References

1. Gel, Y. R., Miao, W. and Gastwirth, J. L. (2007). Robust directed tests of normality against heavy-tailed alternatives. Computational Statistics & Data Analysis, 51(5), 2734-2746. [Source](https://doi.org/10.1016/j.csda.2006.08.022)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GMGNormalityGofStatistic,
)


test_statistic = GMGNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
