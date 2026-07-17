# Kuiper test for beta distribution

## Description
Performs Kuiper goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic combines the largest positive and negative empirical distribution deviations.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperBetaGofStatistic,
)


test_statistic = KuiperBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes

$$ V = D^+ + D^- $$

where $D^+$ and $D^-$ are the one-sided deviations between empirical plotting positions and beta CDF values.

## Author(s)
Alexey Mironov

## References
Kuiper, N.H. (1960): Tests concerning random points on a circle. - Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A, vol. 63, pp. 38-47.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperBetaGofStatistic,
)


test_statistic = KuiperBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
