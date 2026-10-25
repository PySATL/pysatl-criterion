# Anderson-Darling test for normality

**Class:** `AndersonDarlingNormalityGofStatistic`

**Code:** `AD_NORMALITY_GOODNESS_OF_FIT`

Anderson-Darling statistic for the normal location-scale family.

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

A2 = -n - sum((2*i-1)*(log(Phi(z_(i))) +
log(1-Phi(z_(n+1-i)))))/n, where z=(x-mean(x))/s1.
This is the unadjusted fitted-normal statistic. Log-CDF and log-survival
functions avoid rounding probabilities to zero or one.

## References

[1] Anderson, T. W. and Darling, D. A. (1954). A Test of Goodness
   of Fit. Journal of the American Statistical Association, 49(268),
   765-769. https://doi.org/10.1080/01621459.1954.10501232
[2] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
   Statistics with Unknown Parameters. The Annals of Statistics,
   4(2), 357-369. https://doi.org/10.1214/aos/1176343411

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import AndersonDarlingNormalityGofStatistic

statistic = AndersonDarlingNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
