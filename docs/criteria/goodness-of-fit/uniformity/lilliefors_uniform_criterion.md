# Lilliefors test for uniformity

## Description
Performs the Lilliefors-type goodness-of-fit statistic for the hypothesis of uniformity on the interval $[a, b]$.
The implementation reuses the common Lilliefors statistic calculation with the cumulative distribution function values of the reference uniform distribution.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsTestUniformGofStatistic,
)


test_statistic = LillieforsTestUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sample is sorted and transformed with

$$ F_0(x) = \frac{x - a}{b - a}. $$

The transformed values are passed to the common Lilliefors statistic implementation.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Lilliefors, H.W. (1967): On the Kolmogorov-Smirnov test for normality with mean and variance unknown. - Journal of the American Statistical Association, vol. 62, pp. 399-402.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsTestUniformGofStatistic,
)


test_statistic = LillieforsTestUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
