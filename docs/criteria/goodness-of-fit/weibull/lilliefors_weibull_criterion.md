# Lilliefors test for Weibull distribution

## Description
Performs Lilliefors goodness-of-fit test for the hypothesis that the sample comes from a Weibull distribution.
The implementation uses the Weibull distribution utilities from `pysatl_criterion.core.distributions.weibull` where applicable.

Hypothesis of Weibull Distribution
The null hypothesis is that the data comes from a Weibull distribution with parameters `a` and `k`.
The observations passed to `execute_statistic` should be positive for statistics that use logarithms or Weibull probability plots.

Test Statistic
The statistic is based on a Kolmogorov-Smirnov type statistic computed from Weibull CDF values.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsWeibullGofStatistic,
)


test_statistic = LillieforsWeibullGofStatistic(a=1, k=2)
statistic_result = test_statistic.execute_statistic([0.42, 0.65, 0.88, 1.12, 1.43, 1.76, 2.05, 2.44, 2.91, 3.37])
print(statistic_result)
```

## Arguments
`a` - Weibull distribution parameter. Default value is `1`.

`k` - Weibull distribution parameter. Default value is `1` for most statistics; `KolmogorovSmirnovWeibullGofStatistic` defaults to `5`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation evaluates the Lilliefors statistic for the supplied observations. For EDF-based statistics, observations are transformed with the Weibull cumulative distribution function. For spacing, probability plot, Laplace-transform, and moment-style statistics, the implementation follows the corresponding formulas in `pysatl_criterion.statistics.goodness_of_fit.weibull`.

## Author(s)
Alexey Mironov

## References
The statistic follows the implementation in `pysatl_criterion.statistics.goodness_of_fit.weibull`.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsWeibullGofStatistic,
)


test_statistic = LillieforsWeibullGofStatistic(a=1, k=2)
statistic_result = test_statistic.execute_statistic([0.42, 0.65, 0.88, 1.12, 1.43, 1.76, 2.05, 2.44, 2.91, 3.37])
print(statistic_result)
```
