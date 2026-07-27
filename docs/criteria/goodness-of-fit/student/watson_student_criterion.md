# Watson test for Student distribution

## Description
Performs Watson goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The Watson statistic is a centered modification of the Cramer-von Mises statistic.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`, location parameter `loc`, and positive scale parameter `scale`.

Test Statistic
The statistic removes the squared mean deviation of the probability-transformed observations from the Cramer-von Mises statistic.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonStudentGofStatistic,
)


test_statistic = WatsonStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`loc` - location parameter of the Student's t-distribution. Default value is `0`.

`scale` - positive scale parameter of the Student's t-distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes Cramer-von Mises terms from Student CDF values and subtracts the Watson centering correction

$$ n\left(\bar F_0 - \frac{1}{2}\right)^2. $$

## Author(s)
Dmitriy Rusanov, Alexey Mironov

## References
Watson, G.S. (1961): Goodness-of-fit tests on a circle. - Biometrika, vol. 48, pp. 109-114.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonStudentGofStatistic,
)


test_statistic = WatsonStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
