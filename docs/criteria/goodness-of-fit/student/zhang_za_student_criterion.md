# Zhang Za test for Student distribution

## Description
Performs Zhang Za goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The statistic uses weighted logarithms of Student CDF and survival values.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`, location parameter `loc`, and positive scale parameter `scale`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangZaStudentGofStatistic,
)


test_statistic = ZhangZaStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`loc` - location parameter of the Student's t-distribution. Default value is `0`.

`scale` - positive scale parameter of the Student's t-distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes observations, evaluates clipped Student CDF values, and computes Zhang's $Z_A$ statistic from weighted log-CDF and log-survival terms.

## Author(s)
Alexey Mironov

## References
Zhang, J. (2002): Powerful goodness-of-fit tests based on the likelihood ratio. - Journal of the Royal Statistical Society, Series B, vol. 64, pp. 281-294.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangZaStudentGofStatistic,
)


test_statistic = ZhangZaStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
