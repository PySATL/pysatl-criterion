# Hosking1 test for normality

Hosking L-moment normality statistic without trimming.

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
The upper tail of the statistic defines rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Local quadratic discrepancy of L-moment ratios with trim t=0.
For r=2,3,4, L_r=sum_i x_(i)*sum_{k=0..r-1}((-1)**k*
C(r-1,k)*C(i-1,r+t-1-k)*C(n-i,t+k))/(r*C(n,r+2*t)).
Return (L3/L2)**2/v3+(L4/L2-mu4)**2/v4. The constants
(mu4,v3,v4) are the three historical rows in the implementation
for n<=25, 25<n<=50 and n>50. They are not a consistent
asymptotic covariance sequence. A primary source for this exact
omnibus formula and these rows was not located; the L-moment
papers alone do not validate this test. Treat it as a local
statistic and simulate its null law at the actual n. L2 must be positive.

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import Hosking1NormalityGofStatistic

statistic = Hosking1NormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
