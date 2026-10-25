# Robust Jarque-Bera test for log-normal data

**Class:** `RobustJarqueBeraLogNormalGofStatistic`

**Code:** `RJB_LOGNORMAL_GOODNESS_OF_FIT`

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

The implementation computes `RobustJarqueBeraNormalityGofStatistic` on `y = log(rvs)` with an empty
normal-family hypothesis. Thus its formula, sample-size limits, normalization
and rejection tail are exactly those of the normality statistic at its default
settings. Constructor-specific normality options are not exposed by this wrapper.
The identity was checked on normal, nonnormal and tied logarithmic samples and
under transformations `X -> exp(a) * X**b`, b > 0.

## Underlying normality definition

**In this section, x denotes y = log(X), not the original positive sample.**
The normal-family hypothesis and dispersion conventions below concern these
log observations. See also the [normality source page](../normality/robust_jarque_bera_normality_criterion.md).


H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
Location and positive scale cancel, allowing N(0,1) null simulation.
Every simulated sample must go through execute_statistic again.
Large values of the statistic define rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Let J=sqrt(pi/2)*mean(abs(x-median(x))). Return
n*(m3/J**3)**2/6 + n*(m4/J**4-3)**2/64.
Moments are centered at the sample mean, whereas J uses the median.
The reference identifies this robust-scale modification; small-sample
chi-square calibration is not justified.

## References

[1] Gel, Y. R. and Gastwirth, J. L. (2008).
   A robust modification of the Jarque-Bera test of normality.
   Economics Letters, 99(1), 30-32.
   https://doi.org/10.1016/j.econlet.2007.05.022


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
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import RobustJarqueBeraLogNormalGofStatistic

parameters = LogNormal.DEFAULT.parse({})
statistic = RobustJarqueBeraLogNormalGofStatistic(parameters)
sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1])
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
