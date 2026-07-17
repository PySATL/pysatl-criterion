# Anderson-Darling test for Student distribution

## Description
Performs Anderson-Darling goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The test gives additional weight to discrepancies in the distribution tails.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`, location parameter `loc`, and positive scale parameter `scale`.

Test Statistic
The statistic is computed from the log cumulative distribution function and log survival function of the reference Student distribution.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingStudentGofStatistic,
)


test_statistic = AndersonDarlingStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`loc` - location parameter of the Student's t-distribution. Default value is `0`.

`scale` - positive scale parameter of the Student's t-distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes the ordered observations and evaluates Student `logcdf` and `logsf` values.
These values are passed to the common Anderson-Darling statistic implementation.

## Author(s)
Alexey Mironov

## References
Anderson, T.W. and Darling, D.A. (1952): Asymptotic theory of certain goodness of fit criteria based on stochastic processes. - Annals of Mathematical Statistics, vol. 23, pp. 193-212.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingStudentGofStatistic,
)


test_statistic = AndersonDarlingStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
