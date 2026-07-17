# Ratio test for beta distribution

## Description
Performs ratio goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares the sample ratio of geometric mean to arithmetic mean with the theoretical beta distribution ratio.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RatioBetaGofStatistic,
)


test_statistic = RatioBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sample statistic is the ratio of the geometric mean to the arithmetic mean.
The theoretical ratio uses

$$ E[X] = \frac{\alpha}{\alpha + \beta} $$

and

$$ E[\log X] = \psi(\alpha) - \psi(\alpha + \beta), $$

where $\psi$ is the digamma function.

## Author(s)
Alexey Mironov

## References
The statistic follows the implementation in `pysatl_criterion.statistics.goodness_of_fit.beta`.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RatioBetaGofStatistic,
)


test_statistic = RatioBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
