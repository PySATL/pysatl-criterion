# Lilliefors test for beta distribution

## Description
Performs Lilliefors-type goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The implementation reuses the common Lilliefors statistic calculation with beta cumulative distribution function values.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsTestBetaGofStatistic,
)


test_statistic = LillieforsTestBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sample is sorted and transformed with the reference beta cumulative distribution function.
The transformed values are passed to the common Lilliefors statistic implementation.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References
Lilliefors, H.W. (1967): On the Kolmogorov-Smirnov test for normality with mean and variance unknown. - Journal of the American Statistical Association, vol. 62, pp. 399-402.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsTestBetaGofStatistic,
)


test_statistic = LillieforsTestBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
