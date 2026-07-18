# Spiegelhalter test for normality

## Description
Performs the Spiegelhalter goodness-of-fit test for the hypothesis of normality.
The null hypothesis is that the sample comes from a normal distribution.

Hypothesis of Normality
The hypothesis of normality refers to the null hypothesis that the data comes from a normal distribution. In the implementation, the statistic is computed from the sample passed to `execute_statistic`.

Test Statistic
The statistic is based on sample range, mean absolute deviation, and sample variance.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SpiegelhalterNormalityGofStatistic,
)


test_statistic = SpiegelhalterNormalityGofStatistic()
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```

## Arguments
`mean` - reference normal mean. Default value is `0`.

`var` - reference normal variance. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation evaluates the Spiegelhalter statistic for the supplied observations. Large or small values should be interpreted according to the statistic alternative used by the class implementation.

## References
The statistic follows the implementation in `pysatl_criterion.statistics.goodness_of_fit.normal`.

## Author(s)
Alexey Mironov

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SpiegelhalterNormalityGofStatistic,
)


test_statistic = SpiegelhalterNormalityGofStatistic()
statistic_result = test_statistic.execute_statistic([-1.21, -0.83, -0.52, -0.31, -0.08, 0.14, 0.29, 0.47, 0.68, 0.91, 1.16, 1.43])
print(statistic_result)
```
