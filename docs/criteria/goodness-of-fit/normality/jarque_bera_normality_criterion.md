# Jarque-Bera test for normality

**Class:** `JBNormalityGofStatistic`

**Code:** `JB_NORMALITY_GOODNESS_OF_FIT`

Jarque-Bera statistic based on sample skewness and excess kurtosis.

## Methods

execute_statistic(rvs, **kwargs)
    Return one scalar statistic.
hypothesis()
    Report fixed null parameters only.
alternative()
    Report the critical tail.

## Notes

H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
Location and positive scale cancel, allowing N(0,1) null simulation.
Every simulated sample must go through execute_statistic again.
Large values of the statistic define rejection. Use at least 2
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

JB = n*(g1**2/6 + (g2-3)**2/24), where
g1=m3/m2**1.5, g2=m4/m2**2, and mk=mean((x-mean(x))**k).

## References

[1] Jarque, C. M. and Bera, A. K. (1987). A Test for Normality of
   Observations and Regression Residuals. International Statistical
   Review, 55(2), 163-172. https://doi.org/10.2307/1403192

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import JBNormalityGofStatistic

statistic = JBNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
