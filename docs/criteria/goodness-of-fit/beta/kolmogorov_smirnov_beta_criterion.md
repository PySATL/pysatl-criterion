# Kolmogorov-Smirnov test for beta distribution

## Description
Performs Kolmogorov-Smirnov goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The Kolmogorov-Smirnov test compares the empirical distribution function with the theoretical cumulative distribution function of the beta distribution.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

Test Statistic
The statistic is based on the maximum distance between the empirical distribution function and the beta cumulative distribution function.

## Usage
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovBetaGofStatistic,
)


test_statistic = KolmogorovSmirnovBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`alternative_type` - selects the CDF distance: `TWO_TAILED` computes `max(D+, D-)`, `RIGHT` computes `D+`, and `LEFT` computes `D-`. All three reject in the right tail of the statistic distribution.

`mode` - retained for compatibility, default `auto`; it does not affect the distance or compute a p-value here.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
The observations are sorted and transformed with the reference beta cumulative distribution function

$$ F_0(x) = I_x(\alpha, \beta), \quad 0 \le x \le 1. $$

The transformed values are passed to the common Kolmogorov-Smirnov statistic implementation.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References

The reference is for the classical continuous-null KS statistic; a specified Beta CDF is substituted here. Both shapes are known. For unknown shapes use the fitted-Beta Lilliefors-type class.

N. Smirnov (1948), "Table for Estimating the Goodness of Fit
   of Empirical Distributions", Ann. Math. Statist. 19, 279-281.
   https://doi.org/10.1214/aoms/1177730256

## Examples
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovBetaGofStatistic,
)


test_statistic = KolmogorovSmirnovBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
