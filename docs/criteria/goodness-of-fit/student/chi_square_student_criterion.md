# Chi-square test for Student distribution

## Description
Performs chi-square goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The implementation bins standardized observations using equiprobable Student quantile bins.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`, location parameter `loc`, and positive scale parameter `scale`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ChiSquareStudentGofStatistic,
)


test_statistic = ChiSquareStudentGofStatistic(df=5, loc=0, scale=1, n_bins=4)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`loc` - location parameter of the Student's t-distribution. Default value is `0`.

`scale` - positive scale parameter of the Student's t-distribution. Default value is `1`.

`n_bins` - number of equiprobable Student quantile bins. Default value is `10`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes observations, builds bin edges from Student quantiles, and compares observed bin counts with equal expected counts.

## Author(s)
Dmitriy Rusanov, Alexey Mironov

## References
Pearson, K. (1900): On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. - Philosophical Magazine, vol. 50, pp. 157-175.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ChiSquareStudentGofStatistic,
)


test_statistic = ChiSquareStudentGofStatistic(df=5, loc=0, scale=1, n_bins=4)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
