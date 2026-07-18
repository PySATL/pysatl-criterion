# Moran test for gamma distribution

## Description
Performs Moran log-spacing goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic is based on logarithms of spacings after the gamma probability integral transform.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MoranGammaGofStatistic,
)


test_statistic = MoranGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For spacings $D_i$ between consecutive gamma CDF values, the implementation returns

$$ M = -\sum_i \log(nD_i). $$

Spacings must be strictly positive.

## Author(s)
Sergey Golovachev, Alexey Mironov

## References
Moran, P.A.P. (1951): The random division of an interval. - Journal of the Royal Statistical Society, Series B, vol. 13, pp. 147-150.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MoranGammaGofStatistic,
)


test_statistic = MoranGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
