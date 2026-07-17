# Cressie-Read test for gamma distribution

## Description
Performs Cressie-Read power-divergence goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic generalizes Pearson chi-squared and likelihood-ratio statistics.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CressieReadGammaGofStatistic,
)


test_statistic = CressieReadGammaGofStatistic(power=2 / 3, bins=4, alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`power` - Cressie-Read power-divergence parameter. Default value is `2 / 3`.

`bins` - number of equiprobable gamma bins. Default value is `8`.

`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation uses equiprobable gamma quantile bins and passes observed and expected counts to the common chi-squared statistic implementation with the selected power-divergence parameter.

## Author(s)
Alexey Mironov

## References
Cressie, N. and Read, T.R.C. (1984): Multinomial goodness-of-fit tests. - Journal of the Royal Statistical Society, Series B, vol. 46, pp. 440-464.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CressieReadGammaGofStatistic,
)


test_statistic = CressieReadGammaGofStatistic(power=2 / 3, bins=4, alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
