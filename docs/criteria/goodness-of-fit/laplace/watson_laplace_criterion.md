# Watson test for Laplace distribution

## Description
Performs Watson goodness-of-fit test for the hypothesis that the sample comes from a Laplace distribution.
The Watson statistic is a centered modification of the Cramer-von Mises statistic.

Hypothesis of Laplace Distribution
The null hypothesis is that the data comes from a Laplace distribution with location parameter `t` and positive scale parameter `s`.

Test Statistic
The statistic removes the squared mean deviation of the probability-transformed observations from the Cramer-von Mises statistic.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonLaplaceGofStatistic,
)


test_statistic = WatsonLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```

## Arguments
`t` - location parameter of the Laplace distribution. Default value is `0.0`.

`s` - positive scale parameter of the Laplace distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes Cramer-von Mises terms from Laplace CDF values and subtracts the Watson centering correction.
Large values indicate stronger deviation from the Laplace model.

## Author(s)
Kirill Tahmazidi, Alexey Mironov

## References
Watson, G.S. (1961): Goodness-of-fit tests on a circle. - Biometrika, vol. 48, pp. 109-114.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonLaplaceGofStatistic,
)


test_statistic = WatsonLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```
