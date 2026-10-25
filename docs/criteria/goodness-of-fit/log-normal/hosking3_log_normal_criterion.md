# Hosking3 test for log-normal data

**Class:** `Hosking3LogNormalGofStatistic`

**Code:** `HOSKING3_LOGNORMAL_GOODNESS_OF_FIT`

## Hypothesis and inputs

This is a **log-normal family/shape** statistic: log(X) is normal with unknown
location and positive scale. Use `LogNormalDistributionDescriptor.DEFAULT.parse({})`.
An empty `SHAPE_SCALE` schema is also accepted. All observations must be finite,
real, strictly positive and one-dimensional. Active masks and undefined
constant-sample contrasts raise `ValueError`; minimum sizes are specified below.

For compatibility, fully specified `mu, s` or `s, scale` coordinates are also
accepted and preserved by `hypothesis()`. They do **not** enter this shape
statistic: it cannot detect a wrong location/scale within the log-normal family.
Use a fixed-CDF EDF or frequency criterion when those particular values are
part of the hypothesis. Partial parameter sets are rejected.

## Logarithmic reduction

The implementation computes `Hosking3NormalityGofStatistic` on `y = log(rvs)` with an empty
normal-family hypothesis. Thus its formula, sample-size limits, normalization
and rejection tail are exactly those of the normality statistic at its default
settings. Constructor-specific normality options are not exposed by this wrapper.
The identity was checked on normal, nonnormal and tied logarithmic samples and
under transformations `X -> exp(a) * X**b`, b > 0.

## Underlying normality definition

**In this section, x denotes y = log(X), not the original positive sample.**
The normal-family hypothesis and dispersion conventions below concern these
log observations. See also the [normality source page](../normality/hosking3_normality_criterion.md).


H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
Location and positive scale cancel, allowing N(0,1) null simulation.
Every simulated sample must go through execute_statistic again.
Large values of the statistic define rejection. Use at least 8
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Local quadratic discrepancy of L-moment ratios with trim t=2.
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


## Rejection and calibration

Rejection tail: **right**. A statistic value alone is not a p-value.
For the composite null, simulation can use log-normal mu=0, s=1 because
log-location and log-scale are removed by the statistic. Every replicate must
call `execute_statistic` again. This does not make every directed shape test
omnibus. Signed two-tail tests require both null quantiles, without assuming
symmetry. Legacy attribution qualifications carry over from normality.

## Examples

```python
import numpy as np
from pysatl_criterion.distribution import LogNormalDistributionDescriptor as LogNormal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import Hosking3LogNormalGofStatistic

parameters = LogNormal.DEFAULT.parse({})
statistic = Hosking3LogNormalGofStatistic(parameters)
sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1])
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
