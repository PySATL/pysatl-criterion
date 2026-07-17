# Greenwood test for gamma distribution

## Description
Performs Greenwood spacing goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic measures squared spacings after transforming observations by the gamma cumulative distribution function.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GreenwoodGammaGofStatistic,
)


test_statistic = GreenwoodGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes spacings $D_i$ between consecutive gamma CDF values after adding 0 and 1, and returns

$$ G = \sum_i D_i^2. $$

## Author(s)
Alexey Mironov

## References
Greenwood, M. (1946): The statistical study of infectious diseases. - Journal of the Royal Statistical Society, Series A, vol. 109, pp. 85-110.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GreenwoodGammaGofStatistic,
)


test_statistic = GreenwoodGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
