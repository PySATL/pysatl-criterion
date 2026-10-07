# SF test for normality

Shapiro-Francia ``W_prime`` statistic for the normal family.

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
The lower tail of the statistic defines rejection. Use at least 3
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Return corr(x_(i), Phi**-1((i-3/8)/(n+1/4)))**2.
This Shapiro-Francia approximation uses Blom scores in place of
exact expected normal order statistics.

## References

[1] Shapiro, S. S. and Francia, R. S. (1972). An Approximate
   Analysis of Variance Test for Normality. Journal of the American
   Statistical Association, 67(337), 215-216.
   https://doi.org/10.1080/01621459.1972.10481232
[2] Weisberg, S. and Bingham, C. (1975). An Approximate Analysis of
   Variance Test for Non-Normality Suitable for Machine Calculation.
   Technometrics, 17(1), 133-134.
   https://doi.org/10.1080/00401706.1975.10489283

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import SFNormalityGofStatistic

statistic = SFNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
