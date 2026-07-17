# Kuiper test for gamma distribution

## Description
Performs Kuiper goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic combines the largest positive and negative empirical distribution deviations.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperGammaGofStatistic,
)


test_statistic = KuiperGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes

$$ V = D^+ + D^- $$

where $D^+$ and $D^-$ are one-sided deviations between empirical plotting positions and gamma CDF values.

## Author(s)
Alexey Mironov

## References
Kuiper, N.H. (1960): Tests concerning random points on a circle. - Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A, vol. 63, pp. 38-47.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperGammaGofStatistic,
)


test_statistic = KuiperGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
