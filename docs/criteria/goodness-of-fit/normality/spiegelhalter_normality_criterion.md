# Spiegelhalter test for normality

**Class:** `SpiegelhalterNormalityGofStatistic`

**Code:** `SH_NORMALITY_GOODNESS_OF_FIT`

Spiegelhalter statistic for the normal family.

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
Both tails of the statistic define rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

The publication was identified, but the exact finite-sample
implementation is not fully certified against its primary text.

Let U=(max(x)-min(x))/s1, G=sum(abs(x-mean(x)))/
(s1*sqrt(n*(n-1))), and c_n=Gamma(n+1)**(1/(n-1))/(2*n).
Return ((c_n*U)**(-(n-1))+G**(-(n-1)))**(1/(n-1)).
Log-gamma and logaddexp evaluate this expression without overflowing.
Both critical tails are retained; the full primary formula and
its tail convention could not be independently checked in this review.

## References

[1] Spiegelhalter, D. J. (1977). "A test for normality against
   symmetric alternatives." Biometrika, 64(2), 415-418.
   https://doi.org/10.1093/biomet/64.2.415

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import SpiegelhalterNormalityGofStatistic

statistic = SpiegelhalterNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
