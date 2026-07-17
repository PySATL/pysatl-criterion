# Kolmogorov-Smirnov test for Laplace distribution

## Description
Performs Kolmogorov-Smirnov goodness-of-fit test for the hypothesis that the sample comes from a Laplace distribution.
The statistic compares the empirical distribution function with the theoretical Laplace cumulative distribution function.

Hypothesis of Laplace Distribution
The null hypothesis is that the data comes from a Laplace distribution with location parameter `t` and positive scale parameter `s`.

Test Statistic
The statistic is based on the maximum distance between the empirical distribution function and the reference Laplace cumulative distribution function.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovLaplaceGofStatistic,
)


test_statistic = KolmogorovSmirnovLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```

## Arguments
`t` - location parameter of the Laplace distribution. Default value is `0.0`.

`s` - positive scale parameter of the Laplace distribution. Default value is `1.0`.

`alternative_type` - alternative hypothesis type used by the Kolmogorov-Smirnov statistic.

`mode` - calculation mode passed to the Kolmogorov-Smirnov statistic. Default value is `auto`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation evaluates the Laplace cumulative distribution function

$$ F_0(x) = F_{\mathrm{Laplace}(t, s)}(x) $$

for the ordered observations and passes these values to the common Kolmogorov-Smirnov statistic implementation.

## Author(s)
Alexey Mironov

## References
Kolmogorov, A.N. (1933): Sulla determinazione empirica di una legge di distribuzione. - Giornale dell'Istituto Italiano degli Attuari, vol. 4, pp. 83-91.

Smirnov, N.V. (1948): Table for estimating the goodness of fit of empirical distributions. - Annals of Mathematical Statistics, vol. 19, pp. 279-281.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovLaplaceGofStatistic,
)


test_statistic = KolmogorovSmirnovLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```
