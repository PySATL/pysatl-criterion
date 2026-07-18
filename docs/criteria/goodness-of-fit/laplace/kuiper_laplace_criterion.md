# Kuiper test for Laplace distribution

## Description
Performs Kuiper goodness-of-fit test for the hypothesis that the sample comes from a Laplace distribution.
The statistic combines the largest positive and negative empirical distribution deviations.

Hypothesis of Laplace Distribution
The null hypothesis is that the data comes from a Laplace distribution with location parameter `t` and positive scale parameter `s`.

Test Statistic
The statistic is based on the sum of one-sided empirical distribution discrepancies after the Laplace probability integral transform.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperLaplaceGofStatistic,
)


test_statistic = KuiperLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```

## Arguments
`t` - location parameter of the Laplace distribution. Default value is `0.0`.

`s` - positive scale parameter of the Laplace distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes

$$ V = D^+ + D^- $$

where $D^+$ and $D^-$ are one-sided deviations between empirical plotting positions and Laplace CDF values.

## Author(s)
Kirill Tahmazidi, Alexey Mironov

## References
Kuiper, N.H. (1960): Tests concerning random points on a circle. - Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A, vol. 63, pp. 38-47.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperLaplaceGofStatistic,
)


test_statistic = KuiperLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```
