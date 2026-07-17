# Censored Stein test for uniformity

## Description
Performs a censored Stein-type U-statistic test for the hypothesis of uniformity on the interval $[a, b]$.
When no censoring is supplied, the implementation falls back to `SteinUniformGofStatistic`.

Hypothesis of Uniformity
The null hypothesis is that the observed sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CensoredSteinUniformGofStatistic,
)


test_statistic = CensoredSteinUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic(
    [0.12, 0.25, 0.41, 0.53, 0.77, 0.91],
    censoring_indices=[0, 0, 0, 0, 0, 0],
)
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

`censoring_indices` - binary array where `1` indicates a censored observation and `0` indicates an uncensored observation.

## Details
The implementation standardizes observations to $[0, 1]$.
For censored data, it estimates survival with a Kaplan-Meier-style estimator and computes a weighted pairwise Stein kernel statistic over uncensored observations.

## Author(s)
Alexey Mironov

## References
Stein, C. (1972): A bound for the error in the normal approximation to the distribution of a sum of dependent random variables. - Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability, vol. 2, pp. 583-602.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CensoredSteinUniformGofStatistic,
)


test_statistic = CensoredSteinUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic(
    [0.12, 0.25, 0.41, 0.53, 0.77, 0.91],
    censoring_indices=[0, 0, 0, 0, 0, 0],
)
print(statistic_result)
```
