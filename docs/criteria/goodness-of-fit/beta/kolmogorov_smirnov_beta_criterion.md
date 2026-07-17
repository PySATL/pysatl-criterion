# Kolmogorov-Smirnov test for beta distribution

## Description
Performs Kolmogorov-Smirnov goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The Kolmogorov-Smirnov test compares the empirical distribution function with the theoretical cumulative distribution function of the beta distribution.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

Test Statistic
The statistic is based on the maximum distance between the empirical distribution function and the beta cumulative distribution function.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovBetaGofStatistic,
)


test_statistic = KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`alternative_type` - alternative hypothesis type used by the Kolmogorov-Smirnov statistic.

`mode` - calculation mode passed to the Kolmogorov-Smirnov statistic. Default value is `auto`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The observations are sorted and transformed with the reference beta cumulative distribution function

$$ F_0(x) = I_x(\alpha, \beta), \quad 0 \le x \le 1. $$

The transformed values are passed to the common Kolmogorov-Smirnov statistic implementation.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References
Kolmogorov, A.N. (1933): Sulla determinazione empirica di una legge di distribuzione. - Giornale dell'Istituto Italiano degli Attuari, vol. 4, pp. 83-91.

Smirnov, N.V. (1948): Table for estimating the goodness of fit of empirical distributions. - Annals of Mathematical Statistics, vol. 19, pp. 279-281.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovBetaGofStatistic,
)


test_statistic = KolmogorovSmirnovBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
