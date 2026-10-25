# Cabana-Cabana2 test for log-normal data

**Class:** `CabanaCabana2LogNormalGofStatistic`

**Code:** `CC2_LOGNORMAL_GOODNESS_OF_FIT`

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

The implementation computes `CabanaCabana2NormalityGofStatistic` on `y = log(rvs)` with an empty
normal-family hypothesis. Thus its formula, sample-size limits, normalization
and rejection tail are exactly those of the normality statistic at its default
settings. Constructor-specific normality options are not exposed by this wrapper.
The identity was checked on normal, nonnormal and tied logarithmic samples and
under transformations `X -> exp(a) * X**b`, b > 0.

## Underlying normality definition

**In this section, x denotes y = log(X), not the original positive sample.**
The normal-family hypothesis and dispersion conventions below concern these
log observations. See also the [normality source page](../normality/cabana_cabana2_normality_criterion.md).


H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
Location and positive scale cancel, allowing N(0,1) null simulation.
Every simulated sample must go through execute_statistic again.
Large values of the statistic define rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Use the l=5 process in Cabana and Cabana, equation (8).
For z=(x-mean(x))/s1, h_j=He_j/sqrt(j!), H_j=sum(h_j(z))/sqrt(n),
Q(t)=sum_{j=2..5}((sqrt(j/(j-1))*h_(j-2)(t)+h_j(t))*H_(j+3)).
Set P(t)=H_3+t*H_4+Q(t). Return
sup_t abs(Phi(t)*H_4-phi(t)*P(t)). Evaluate every real stationary
point (roots of H_4+t*P-P') and both limits at infinity.
The H_8 term occurs once. This is the truncated-process supremum;
polynomial root solving is subject to float64 accuracy.

## References

[1] Cabaña, A. and Cabaña, E. M. (2003).
   Tests of Normality Based on Transformed Empirical Processes.
   Methodology and Computing in Applied Probability, 5, 309-335.
   https://doi.org/10.1023/A:1026235220018


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
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import CabanaCabana2LogNormalGofStatistic

parameters = LogNormal.DEFAULT.parse({})
statistic = CabanaCabana2LogNormalGofStatistic(parameters)
sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1])
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
