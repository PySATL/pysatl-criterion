# Anderson-Darling test for Laplace distribution

## Description
Performs Anderson-Darling goodness-of-fit test for the hypothesis that the sample comes from a Laplace distribution.
The test gives additional weight to discrepancies in the distribution tails.

Hypothesis of Laplace Distribution
The null hypothesis is that the data comes from a Laplace distribution with location parameter `t` and positive scale parameter `s`.

Test Statistic
The statistic is computed from the log cumulative distribution function and log survival function of the reference Laplace distribution.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingLaplaceGofStatistic,
)


test_statistic = AndersonDarlingLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```

## Arguments
`t` - location parameter of the Laplace distribution. Default value is `0.0`.

`s` - positive scale parameter of the Laplace distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation evaluates Laplace `logcdf` and `logsf` values for the ordered sample and passes them to the common Anderson-Darling statistic implementation.
Large values indicate stronger deviation from the Laplace model.

## Author(s)
Alexey Mironov

## References
Anderson, T.W. and Darling, D.A. (1952): Asymptotic theory of certain goodness of fit criteria based on stochastic processes. - Annals of Mathematical Statistics, vol. 23, pp. 193-212.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingLaplaceGofStatistic,
)


test_statistic = AndersonDarlingLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```
