# Kuiper test for Student distribution

## Description
Performs Kuiper goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The statistic combines the largest positive and negative empirical distribution deviations.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`, location parameter `loc`, and positive scale parameter `scale`.

Test Statistic
The statistic is based on the sum of one-sided empirical distribution discrepancies after the Student probability integral transform.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperStudentGofStatistic,
)


test_statistic = KuiperStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`loc` - location parameter of the Student's t-distribution. Default value is `0`.

`scale` - positive scale parameter of the Student's t-distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes

$$ V = D^+ + D^- $$

where $D^+$ and $D^-$ are one-sided deviations between empirical plotting positions and Student CDF values.

## Author(s)
Alexey Mironov

## References
Kuiper, N.H. (1960): Tests concerning random points on a circle. - Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A, vol. 63, pp. 38-47.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperStudentGofStatistic,
)


test_statistic = KuiperStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
