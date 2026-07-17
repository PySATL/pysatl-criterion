# Kolmogorov-Smirnov test for Student distribution

## Description
Performs Kolmogorov-Smirnov goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The statistic compares the empirical distribution function with the theoretical Student cumulative distribution function.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`, location parameter `loc`, and positive scale parameter `scale`.

Test Statistic
The statistic is based on the maximum distance between the empirical distribution function and the reference Student cumulative distribution function.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovStudentGofStatistic,
)


test_statistic = KolmogorovSmirnovStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`loc` - location parameter of the Student's t-distribution. Default value is `0`.

`scale` - positive scale parameter of the Student's t-distribution. Default value is `1`.

`alternative_type` - alternative hypothesis type used by the Kolmogorov-Smirnov statistic.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes the ordered observations as

$$ Z_i = \frac{X_{(i)} - loc}{scale} $$

and evaluates the Student CDF with `df` degrees of freedom.

## Author(s)
Alexey Mironov

## References
Kolmogorov, A.N. (1933): Sulla determinazione empirica di una legge di distribuzione. - Giornale dell'Istituto Italiano degli Attuari, vol. 4, pp. 83-91.

Smirnov, N.V. (1948): Table for estimating the goodness of fit of empirical distributions. - Annals of Mathematical Statistics, vol. 19, pp. 279-281.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovStudentGofStatistic,
)


test_statistic = KolmogorovSmirnovStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
