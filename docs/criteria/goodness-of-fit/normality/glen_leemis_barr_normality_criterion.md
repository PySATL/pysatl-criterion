# Glen-Leemis-Barr test for normality

## Description

Performs the Glen-Leemis-Barr goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on transformed fitted normal probabilities and beta distribution values.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GlenLeemisBarrNormalityGofStatistic,
)


test_statistic = GlenLeemisBarrNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The sample mean and standard deviation with `ddof=1` define a fitted
normal CDF. The ordered probabilities are transformed by their respective
beta CDFs, sorted again, and combined in an Anderson-Darling-style sum.
Large values correspond to the right-tail alternative.

Use at least four finite observations with nonzero sample variance.
The reference describes the order-statistic construction; this class uses
its fitted-normal version, whose null calibration includes estimation.

## References

1. Glen, A. G., Leemis, L. M. and Barr, D. R. (2001). Order statistics in goodness-of-fit testing. IEEE Transactions on Reliability, 50(2), 209-213. [Source](https://doi.org/10.1109/24.963129)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GlenLeemisBarrNormalityGofStatistic,
)


test_statistic = GlenLeemisBarrNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
