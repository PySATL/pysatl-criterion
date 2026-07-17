# Pearson chi-squared test for beta distribution

## Description
Performs Pearson's chi-squared goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The implementation compares observed histogram bin counts with expected counts under the reference beta distribution.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonBetaGofStatistic,
)


test_statistic = Chi2PearsonBetaGofStatistic(alpha=2, beta=5, lambda_=1)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`lambda_` - power-divergence parameter passed to the common chi-squared statistic implementation. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation uses approximately $\sqrt n$ bins on $[0, 1]$.
Expected frequencies are computed from beta CDF differences at the bin edges.

## Author(s)
Alexey Mironov

## References
Pearson, K. (1900): On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. - Philosophical Magazine, vol. 50, pp. 157-175.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonBetaGofStatistic,
)


test_statistic = Chi2PearsonBetaGofStatistic(alpha=2, beta=5, lambda_=1)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
