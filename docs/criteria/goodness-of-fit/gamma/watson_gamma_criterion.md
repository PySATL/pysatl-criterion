# Watson test for gamma distribution

## Description
Performs Watson goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The Watson statistic is a centered modification of the Cramer-von Mises statistic.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonGammaGofStatistic,
)


test_statistic = WatsonGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes Cramer-von Mises terms from gamma CDF values and subtracts the Watson centering correction.
Large values indicate stronger deviation from the gamma model.

## Author(s)
Alexey Mironov

## References
Watson, G.S. (1961): Goodness-of-fit tests on a circle. - Biometrika, vol. 48, pp. 109-114.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonGammaGofStatistic,
)


test_statistic = WatsonGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
