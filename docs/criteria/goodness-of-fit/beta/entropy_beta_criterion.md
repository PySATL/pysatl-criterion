# Entropy test for beta distribution

## Description
Performs entropy-based goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares a spacing-based sample entropy estimate with the theoretical entropy of the beta distribution.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    EntropyBetaGofStatistic,
)


test_statistic = EntropyBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`m` - optional window size for the spacing-based entropy estimator, passed to `execute_statistic`. Default value is `n // 4`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation uses a Vasicek-style spacing entropy estimator on the sorted sample.
It compares this estimate with the theoretical beta entropy computed from `betaln` and digamma terms.

## Author(s)
Alexey Mironov

## References
Vasicek, O. (1976): A test for normality based on sample entropy. - Journal of the Royal Statistical Society, Series B, vol. 38, pp. 54-59.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    EntropyBetaGofStatistic,
)


test_statistic = EntropyBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
