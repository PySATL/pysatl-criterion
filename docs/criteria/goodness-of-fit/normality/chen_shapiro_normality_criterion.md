# Chen-Shapiro test for normality

## Description

Performs the Chen-Shapiro goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on normalized spacings of ordered observations and expected normal quantiles.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ChenShapiroNormalityGofStatistic,
)


test_statistic = ChenShapiroNormalityGofStatistic()
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

Successive sample spacings are divided by spacings of the normal scores
`norm.ppf((i - 0.375) / (n + 0.25))`. Their average is standardized by
the sample standard deviation with `ddof=1` to form `QH`.
The returned statistic is `sqrt(n) * (1 - QH)`; large values correspond
to the right-tail alternative.

Use at least four finite observations with nonzero sample variance.
The calculation estimates location and scale implicitly from the sample.

## References

1. Chen, L. and Shapiro, S. S. (1995). An alternative test for normality based on normalized spacings. Journal of Statistical Computation and Simulation, 53(3-4), 269-287. [Source](https://doi.org/10.1080/00949659508811711)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ChenShapiroNormalityGofStatistic,
)


test_statistic = ChenShapiroNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
