# Hosking4 test for normality

**Class:** `Hosking4NormalityGofStatistic`

**Code:** `HOSKING4_NORMALITY_GOODNESS_OF_FIT`

Hosking normality statistic with symmetric trimming level 3.

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
Large values of the statistic define rejection. Use at least 10
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Local quadratic discrepancy of L-moment ratios with trim t=3.
For r=2,3,4, L_r=sum_i x_(i)*sum_{k=0..r-1}((-1)**k*
C(r-1,k)*C(i-1,r+t-1-k)*C(n-i,t+k))/(r*C(n,r+2*t)).
Return (L3/L2)**2/v3+(L4/L2-mu4)**2/v4. The constants
(mu4,v3,v4) are the three historical rows in the implementation
for n<=25, 25<n<=50 and n>50. They are not a consistent
asymptotic covariance sequence. A primary source for this exact
omnibus formula and these rows was not located; the L-moment
papers alone do not validate this test. Treat it as a local
statistic and simulate its null law at the actual n. L2 must be positive.

## References

Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions using linear combinations of order statistics. JRSS B, 52(1), 105–124.
https://doi.org/10.1111/j.2517-6161.1990.tb01775.x

Hosking, J. R. M. (2007). Some theory and practical uses of trimmed L-moments. Journal of Statistical Planning and Inference, 137(9), 3024–3039.
https://doi.org/10.1016/j.jspi.2006.12.002

These articles justify the L-moment construction, not the three local rows of calibration constants used here.

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import Hosking4NormalityGofStatistic

statistic = Hosking4NormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
