# Likelihood-ratio test for gamma distribution

## Description
Performs likelihood-ratio goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The test is a G-test variant of the binned chi-squared family.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LikelihoodRatioGammaGofStatistic,
)


test_statistic = LikelihoodRatioGammaGofStatistic(bins=4, alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`bins` - number of equiprobable gamma bins. Default value is `8`.

`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation uses equiprobable gamma quantile bins and the power-divergence statistic with `lambda = 0`.

## Author(s)
Sergey Golovachev, Alexey Mironov

## References
Wilks, S.S. (1935): The likelihood test of independence in contingency tables. - Annals of Mathematical Statistics, vol. 6, pp. 190-196.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LikelihoodRatioGammaGofStatistic,
)


test_statistic = LikelihoodRatioGammaGofStatistic(bins=4, alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
