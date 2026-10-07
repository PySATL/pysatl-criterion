# Zhang Q test for normality

Zhang normality statistic based on a ratio of ordered-sample contrasts.

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
Both tails of the statistic defines rejection. Use at least 8
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

The publication was identified, but the exact finite-sample
implementation is not fully certified against its primary text.

For u_i=Phi**-1((i-3/8)/(n+1/4)), set
q1=mean_{i=2..n}((x_(i)-x_(1))/(u_i-u_1)),
q2=mean_{i=1..n-4}((x_(i+4)-x_(i))/(u_(i+4)-u_i)).
Return log(q1)-log(q2). The four-spacing construction and Blom
approximation are explicit implementation choices. Both contrasts
must be positive. Both critical tails are used for the signed log ratio;
this is the Q component, not a combined Q/Q* p-value procedure.

## References

[1] Zhang, P. (1999).
   Omnibus test of normality using the Q statistic.
   Journal of Applied Statistics, 26(4), 519-528.
   https://doi.org/10.1080/02664769922395

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import ZhangQNormalityGofStatistic

statistic = ZhangQNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
