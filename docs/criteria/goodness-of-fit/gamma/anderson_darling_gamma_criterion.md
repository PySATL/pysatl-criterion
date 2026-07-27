# Anderson-Darling test for gamma distribution

## Description
Performs Anderson-Darling goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The test gives additional weight to discrepancies in the tails.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingGammaGofStatistic,
)


test_statistic = AndersonDarlingGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation evaluates gamma `logcdf` and `logsf` values for the ordered sample and passes them to the common Anderson-Darling statistic implementation.

## Author(s)
Sergey Golovachev, Alexey Mironov

## References
Anderson, T.W. and Darling, D.A. (1952): Asymptotic theory of certain goodness of fit criteria based on stochastic processes. - Annals of Mathematical Statistics, vol. 23, pp. 193-212.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingGammaGofStatistic,
)


test_statistic = AndersonDarlingGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
