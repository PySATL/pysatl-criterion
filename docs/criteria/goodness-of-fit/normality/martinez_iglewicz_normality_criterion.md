# Martinez-Iglewicz test for normality

## Description

Performs the Martinez-Iglewicz goodness-of-fit test for the hypothesis of normality.
The null hypothesis is the normal family with unknown mean and variance.

Hypothesis of Normality
The population mean and variance are free parameters of the normal family.
The statistic estimates them from the sample or eliminates them through
location- and scale-invariant calculations. No mean or variance is fixed in
advance, so `hypothesis().parameters()` returns `{}`. Estimates computed from
observations are not added to that dictionary.

Test Statistic
The statistic is based on a robust scale estimate based on the median and median absolute deviation.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MartinezIglewiczNormalityGofStatistic,
)


test_statistic = MartinezIglewiczNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments

This class takes no constructor arguments.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The numerator uses squared deviations from the sample median. The
denominator is a biweight-style scale estimate using nine times the
median absolute deviation as its tuning scale. Large values correspond
to the right-tail alternative.

Use at least four finite observations, a positive median absolute
deviation, and a nonzero biweight denominator. The current implementation
evaluates the biweight polynomials for all observations without truncating
terms whose standardized absolute deviation is at least one.

## References

1. Martinez, J. and Iglewicz, B. (1981). A test for departure from normality based on a biweight estimator of scale. Biometrika, 68(1), 331-333. [Source](https://doi.org/10.1093/biomet/68.1.331)

## Author(s)

Alexey Mironov

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MartinezIglewiczNormalityGofStatistic,
)


test_statistic = MartinezIglewiczNormalityGofStatistic()
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
