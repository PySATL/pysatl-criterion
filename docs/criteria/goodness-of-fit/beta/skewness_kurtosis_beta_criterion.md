# Skewness-kurtosis test for beta distribution

## Description
Performs skewness-kurtosis goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares sample skewness and kurtosis with the theoretical beta distribution values.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SkewnessKurtosisBetaGofStatistic,
)


test_statistic = SkewnessKurtosisBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes sample skewness and kurtosis and compares them with the theoretical values for the beta distribution.
The squared differences are combined in a Jarque-Bera-like statistic.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References
The statistic follows the implementation in `pysatl_criterion.statistics.goodness_of_fit.beta`.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SkewnessKurtosisBetaGofStatistic,
)


test_statistic = SkewnessKurtosisBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
