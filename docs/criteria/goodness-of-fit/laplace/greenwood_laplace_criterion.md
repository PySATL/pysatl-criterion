# Greenwood test for Laplace distribution

## Description
Performs Greenwood spacing goodness-of-fit test for the hypothesis that the sample comes from a Laplace distribution.
The statistic measures squared spacings after transforming observations by the Laplace cumulative distribution function.

Hypothesis of Laplace Distribution
The null hypothesis is that the data comes from a Laplace distribution with location parameter `t` and positive scale parameter `s`.

Test Statistic
The statistic is based on the sum of squared spacings between consecutive probability-transformed observations.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GreenwoodLaplaceGofStatistic,
)


test_statistic = GreenwoodLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```

## Arguments
`t` - location parameter of the Laplace distribution. Default value is `0.0`.

`s` - positive scale parameter of the Laplace distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes spacings $D_i$ between consecutive Laplace CDF values after adding 0 and 1, and returns

$$ G = \sum_i D_i^2. $$

Large values indicate clustering or uneven spacing in the probability-transformed sample.

## Author(s)
Kirill Tahmazidi, Alexey Mironov

## References
Greenwood, M. (1946): The statistical study of infectious diseases. - Journal of the Royal Statistical Society, Series A, vol. 109, pp. 85-110.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GreenwoodLaplaceGofStatistic,
)


test_statistic = GreenwoodLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```
