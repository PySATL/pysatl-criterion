# Cabana-Cabana1 test for normality

## Description

Performs the Cabana-Cabana1 goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on Hermite-polynomial components and fitted normal CDF/PDF values.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CabanaCabana1NormalityGofStatistic,
)


test_statistic = CabanaCabana1NormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The calculation evaluates a transformed empirical process at the sample
points after standardization by the sample mean and standard deviation
with `ddof=1`. It uses normalized Hermite polynomials through degree
eight, corresponding to the fixed truncation order `l=5`.
The maximum absolute process value gives a right-tail statistic.

Use at least four finite observations with nonzero sample variance.
The normality construction is described in the 2003 paper cited below.

## References

1. Cabaña, A. and Cabaña, E. M. (2003). Tests of Normality Based on Transformed Empirical Processes. Methodology and Computing in Applied Probability, 5, 309-335. [Source](https://doi.org/10.1023/A:1026235220018)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CabanaCabana1NormalityGofStatistic,
)


test_statistic = CabanaCabana1NormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
