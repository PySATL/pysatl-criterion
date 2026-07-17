# Lilliefors test for Student distribution

## Description
Performs Lilliefors-type goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The implementation reuses the common Lilliefors statistic calculation with Student CDF values.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsStudentGofStatistic,
)


test_statistic = LillieforsStudentGofStatistic(df=5)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sorted sample is transformed with the Student cumulative distribution function using `df` degrees of freedom.
The transformed values are passed to the common Lilliefors statistic implementation.

## Author(s)
Dmitriy Rusanov, Alexey Mironov

## References
Lilliefors, H.W. (1967): On the Kolmogorov-Smirnov test for normality with mean and variance unknown. - Journal of the American Statistical Association, vol. 62, pp. 399-402.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsStudentGofStatistic,
)


test_statistic = LillieforsStudentGofStatistic(df=5)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
