# Filliben test for normality

## Description

Performs the Filliben goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on the correlation between ordered sample values and normal order medians.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    FilliNormalityGofStatistic,
)


test_statistic = FilliNormalityGofStatistic()
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

The null leaves normal location and scale unrestricted;
`hypothesis().parameters()` is empty. The statistic is the Pearson
correlation between sorted observations and normal quantiles obtained
from Filliben's approximations to uniform order-statistic medians.
Endpoint plotting positions use the separate formulas in [1].

Correlation eliminates the unknown mean and scale. Smaller values
indicate departures from normality, and `alternative()` returns
`LeftAlternative`. Supply finite, nonconstant observations; two
observations always give a trivial correlation, so use at least three.
The returned value is a correlation coefficient, not a p-value.

## References

1. Filliben, J. J. (1975). The Probability Plot Correlation Coefficient Test for Normality. Technometrics, 17(1), 111-117. [Source](https://doi.org/10.1080/00401706.1975.10489279)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    FilliNormalityGofStatistic,
)


test_statistic = FilliNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
