# Hosking2 test for normality

## Description

Performs the Hosking2 goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on trimmed L-moment ratios for skewness and kurtosis.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Hosking2NormalityGofStatistic,
)


test_statistic = Hosking2NormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The null is the normal family with unknown mean and variance;
`hypothesis().parameters()` is empty. The statistic combines squared
deviations of trimmed L-skewness and L-kurtosis, using order-statistic
weights with trimming level `t=1` from the construction in [1].
The ratios remove location and positive scale.

Supply at least 6 finite, nonconstant observations; estimating the
fourth trimmed L-moment requires `n >= 4 + 2*t`. Reference constants
are selected from three sample-size ranges: up to 25, 26-50, and above
50. The current `alternative()` reports `TwoSidedAlternative` for
this nonnegative quadratic discrepancy. Reference [1] supplies the
trimmed-moment foundation; it does not establish the provenance of
the numerical constants or the exact omnibus implementation here.

## References

1. Elamir, E. A. H. and Seheult, A. H. (2003). Trimmed L-moments. Computational Statistics & Data Analysis, 43(3), 299-314. [Source](https://doi.org/10.1016/S0167-9473(02)00250-5)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Hosking2NormalityGofStatistic,
)


test_statistic = Hosking2NormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
