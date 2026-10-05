# Desgagne-Lafaye test for normality

## Description

Performs the Desgagne-Lafaye goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on logarithmic moment components of standardized observations.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DesgagneLafayeNormalityGofStatistic,
)


test_statistic = DesgagneLafayeNormalityGofStatistic()
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

The null hypothesis is normality with unknown mean and variance;
`hypothesis().parameters()` is an empty dictionary. No distribution
parameters are accepted by the constructor.

The sample is centered and divided by its standard deviation computed
with `ddof=0`. The statistic combines three logarithmic measures of
tail thickness in a quadratic form, as in the Rao score test against
generalized exponential power alternatives [1]. The implementation
evaluates `R_n`, although its historical short code is `DLDMZEPD`.

Large values provide evidence against normality; the class uses a
right-sided alternative. At least four observations and nonzero sample
variance are required. A centered observation equal to zero can produce
a nonfinite result because the current implementation evaluates its
logarithm directly.

## References

1. Desgagne, A., Lafaye de Micheaux, P., and Leblanc, A. (2013). "Test of Normality Against Generalized Exponential Power Alternatives." Communications in Statistics - Theory and Methods, 42(1), 164-190. [Source](https://doi.org/10.1080/03610926.2011.577548)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DesgagneLafayeNormalityGofStatistic,
)


test_statistic = DesgagneLafayeNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic(
    [-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43]
)
print(statistic_result)
```
