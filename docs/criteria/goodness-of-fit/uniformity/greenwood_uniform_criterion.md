# Greenwood test for uniformity

## Description
Performs Greenwood's spacing test for the hypothesis of uniformity on the interval $[a, b]$.
The test is based on the squared spacings between adjacent ordered observations after adding the interval boundaries.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GreenwoodTestUniformGofStatistic,
)


test_statistic = GreenwoodTestUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes the sample to $[0, 1]$, adds boundary points 0 and 1, computes spacings $D_i$, and returns

$$ G = \sum_i D_i^2. $$

Large values are associated with uneven spacings.

## Author(s)
Alexey Mironov

## References
Greenwood, M. (1946): The statistical study of infectious diseases. - Journal of the Royal Statistical Society, Series A, vol. 109, pp. 85-110.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GreenwoodTestUniformGofStatistic,
)


test_statistic = GreenwoodTestUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
