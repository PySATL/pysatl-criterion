# Zhang Q-star test for normality

**Class:** `ZhangQStarNormalityGofStatistic`

**Code:** `ZQS_NORMALITY_GOODNESS_OF_FIT`

Reflected Zhang ``Q*`` normality statistic.

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
Both tails of the statistic define rejection. Use at least 8
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

The publication was identified, but the exact finite-sample
implementation is not fully certified against its primary text.

Return Zhang Q evaluated on the reflected sample -x, using
u_i=Phi**-1((i-3/8)/(n+1/4)), the mean spacing from the minimum
for q1 and mean four-spacing for q2; return log(q1)-log(q2).
Both contrasts must be positive. Both tails of the signed log ratio
are used. This is a reflected component, not combined Q/Q* inference.

## References

[1] Zhang, P. (1999).
   Omnibus test of normality using the Q statistic.
   Journal of Applied Statistics, 26(4), 519-528.
   https://doi.org/10.1080/02664769922395

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import ZhangQStarNormalityGofStatistic

statistic = ZhangQStarNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
