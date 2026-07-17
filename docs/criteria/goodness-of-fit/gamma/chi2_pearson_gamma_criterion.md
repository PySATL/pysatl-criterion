# Pearson chi-squared test for gamma distribution

## Description
Performs Pearson chi-squared goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The implementation bins observations using equiprobable gamma quantile bins.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonGammaGofStatistic,
)


test_statistic = Chi2PearsonGammaGofStatistic(bins=4, alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`bins` - number of equiprobable gamma bins. Default value is `8`.

`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The bin edges are gamma quantiles, so each bin has equal theoretical probability under the null model.
Observed bin counts are compared with equal expected counts.

## Author(s)
Alexey Mironov

## References
Pearson, K. (1900): On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. - Philosophical Magazine, vol. 50, pp. 157-175.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonGammaGofStatistic,
)


test_statistic = Chi2PearsonGammaGofStatistic(bins=4, alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
