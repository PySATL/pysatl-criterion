# Anderson-Darling test for beta distribution

## Description
Performs Anderson-Darling goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The test gives additional weight to discrepancies in the distribution tails.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

Test Statistic
The statistic is computed from the log cumulative distribution function and log survival function of the reference beta distribution.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingBetaGofStatistic,
)


test_statistic = AndersonDarlingBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For ordered observations $X_{(i)}$, the implementation evaluates beta `logcdf` and `logsf` values and computes

$$ A_n^2 = -n - \sum_{i=1}^{n}\frac{2i - 1}{n}\left(\log F_0(X_{(i)}) + \log(1 - F_0(X_{(n+1-i)}))\right). $$

Large values indicate stronger deviation from the beta model.

## Author(s)
Alexey Mironov

## References
Anderson, T.W. and Darling, D.A. (1952): Asymptotic theory of certain goodness of fit criteria based on stochastic processes. - Annals of Mathematical Statistics, vol. 23, pp. 193-212.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingBetaGofStatistic,
)


test_statistic = AndersonDarlingBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
