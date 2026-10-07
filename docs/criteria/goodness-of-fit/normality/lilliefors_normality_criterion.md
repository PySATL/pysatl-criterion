# Lilliefors test for normality

Lilliefors statistic for normality with unknown mean and variance.

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
The upper tail of the statistic defines rejection. Use at least 2
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

D = max_i(i/n-Phi(z_(i)), Phi(z_(i))-(i-1)/n),
where z=(x-mean(x))/s1. Both parameters are re-estimated on every call.
Ordinary fixed-CDF KS tables do not apply.

## References

[1] Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for
   Normality with Mean and Variance Unknown. Journal of the American
   Statistical Association, 62(318), 399-402.
   https://doi.org/10.1080/01621459.1967.10482916

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import LillieforsNormalityGofStatistic

statistic = LillieforsNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
