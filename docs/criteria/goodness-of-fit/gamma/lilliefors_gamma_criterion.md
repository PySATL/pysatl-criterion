# Lilliefors test for gamma distribution

## Description
Performs Lilliefors-type goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The implementation estimates gamma parameters from the sample before computing a Kolmogorov-Smirnov type statistic.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsGammaGofStatistic,
)


test_statistic = LillieforsGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter stored in the statistic hypothesis. Default value is `1.0`.

`beta` - positive rate parameter stored in the statistic hypothesis. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation estimates gamma parameters with

$$ \hat\alpha = \frac{\bar X^2}{S^2}, \quad \hat\theta = \frac{S^2}{\bar X}. $$

The sorted sample is then transformed by the estimated gamma cumulative distribution function.

## Author(s)
Alexey Mironov

## References
Lilliefors, H.W. (1967): On the Kolmogorov-Smirnov test for normality with mean and variance unknown. - Journal of the American Statistical Association, vol. 62, pp. 399-402.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsGammaGofStatistic,
)


test_statistic = LillieforsGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
