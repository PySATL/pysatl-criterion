# Epps-Pulley test for normality

## Description

Performs the Epps-Pulley goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on an empirical characteristic-function statistic.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    EppsPulleyNormalityGofStatistic,
)


test_statistic = EppsPulleyNormalityGofStatistic()
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

The null is the whole normal location-scale family;
`hypothesis().parameters()` is empty. The implementation uses the
sample mean and variance with `ddof=0` to remove location and scale.
Gaussian kernels compare centered observations and all observation
pairs, yielding the closed-form weighted characteristic-function
discrepancy with the smoothing constant fixed by this implementation.

Larger values indicate departures from normality;
`alternative()` returns `RightAlternative`. Supply at least two
finite observations with positive sample variance. Pairwise terms
require quadratic computation in the sample size. No p-value is returned.

## References

1. Epps, T. W. and Pulley, L. B. (1983). A test for normality based on the empirical characteristic function. Biometrika, 70(3), 723-726. [Source](https://doi.org/10.1093/biomet/70.3.723)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    EppsPulleyNormalityGofStatistic,
)


test_statistic = EppsPulleyNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
