# Watson test for beta distribution

## Description
Performs Watson goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The Watson statistic is a centered modification of the Cramer-von Mises statistic.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonBetaGofStatistic,
)


test_statistic = WatsonBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes a Cramer-von Mises statistic from beta CDF values and subtracts the Watson correction term

$$ n\left(\bar F_0 - \frac{1}{2}\right)^2. $$

Large values indicate stronger deviation from the beta model.

## Author(s)
Alexey Mironov

## References
Watson, G.S. (1961): Goodness-of-fit tests on a circle. - Biometrika, vol. 48, pp. 109-114.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonBetaGofStatistic,
)


test_statistic = WatsonBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
