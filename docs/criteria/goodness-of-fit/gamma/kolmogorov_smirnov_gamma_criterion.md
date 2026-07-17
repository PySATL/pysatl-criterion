# Kolmogorov-Smirnov test for gamma distribution

## Description
Performs Kolmogorov-Smirnov goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic compares the empirical distribution function with the theoretical gamma cumulative distribution function.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

Test Statistic
The statistic is based on the maximum distance between the empirical distribution function and the reference gamma cumulative distribution function.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovGammaGofStatistic,
)


test_statistic = KolmogorovSmirnovGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`alternative_type` - alternative hypothesis type used by the Kolmogorov-Smirnov statistic.

`mode` - calculation mode passed to the Kolmogorov-Smirnov statistic. Default value is `auto`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation evaluates the gamma cumulative distribution function using `scale=1 / beta`:

$$ F_0(x) = F_{\Gamma(\alpha, 1 / \beta)}(x). $$

The transformed ordered observations are passed to the common Kolmogorov-Smirnov statistic implementation.

## Author(s)
Alexey Mironov

## References
Kolmogorov, A.N. (1933): Sulla determinazione empirica di una legge di distribuzione. - Giornale dell'Istituto Italiano degli Attuari, vol. 4, pp. 83-91.

Smirnov, N.V. (1948): Table for estimating the goodness of fit of empirical distributions. - Annals of Mathematical Statistics, vol. 19, pp. 279-281.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovGammaGofStatistic,
)


test_statistic = KolmogorovSmirnovGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
