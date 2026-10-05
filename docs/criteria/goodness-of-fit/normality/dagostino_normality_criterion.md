# D'Agostino test for normality

## Description

Performs the D'Agostino goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a transformed linear combination of ordered observations.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DagostinoNormalityGofStatistic,
)


test_statistic = DagostinoNormalityGofStatistic()
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

A linear contrast of the order statistics is divided by the sample
standard deviation computed with `ddof=0` and by `n**2` to form `D`.
This implementation returns
`sqrt(n) * (D - 0.28209479) / 0.02998598`.
Both low and high values correspond to the two-sided alternative.

Use at least four finite observations with nonzero sample variance.
This order-statistic construction is distinct from the skewness-and-
kurtosis D'Agostino-Pearson statistic.

## References

1. D'Agostino, R. B. (1971). An omnibus test of normality for moderate and large size samples. Biometrika, 58(2), 341-348. [Source](https://doi.org/10.1093/biomet/58.2.341)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DagostinoNormalityGofStatistic,
)


test_statistic = DagostinoNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
