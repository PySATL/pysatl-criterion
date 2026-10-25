# KL integral test for log-normal distribution

**Class:** `KLIntegralLogNormalGoFStatistic`

**Code:** `KL_INT_LOGNORMAL_GOODNESS_OF_FIT`

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

## Statistic

At least two positive, nonconstant observations are required. Set
$y_i=\log x_i$, $z_i=(y_i-\bar y)/s_0$ with
$s_0^2=n^{-1}\sum_i(y_i-\bar y)^2$. Define

$$K(r)=\log\left(n^{-1}\sum_i e^{r z_i}\right),\qquad
D(r)=2K(r)-rK'(r),$$
$$\sigma^2(r)=e^{r^2}(r^4-3r^2+4)-(4+r^2),\qquad R(r)=D(r)/\sigma(r).$$

The returned value is $n\int_{-5}^5 R(r)^2\,dr$, computed with adaptive
quadrature split at zero (absolute and relative tolerance 1e-9). It is a
**truncated** integral; no bound on the omitted infinite tails is asserted.

Centering cancels from D exactly. The implementation avoids forming powers
of X and uses log-sum-exp. Near zero it uses cumulants through order 10 and
a positive-coefficient variance series. If m3 = mean(z**3),
$\lim_{r\downarrow0}R(r)=-m_3/\sqrt6$ and
$\lim_{r\to0}R(r)^2=m_3^2/6$. This removes the spurious 0/0 singularity.

## Critical values and source limitations

`calculate_critical_value(n, alpha)` retains the historical numerical
regressions at alpha=0.01 and 0.05. These are **not verified finite-sample
quantiles**. Their published validity range and compatibility with this
truncated implementation have not been established from accessible full text.
Do not use them as a certified significance rule. Invalid n/alpha and
nonpositive or nonfinite regression outputs are rejected;
other significance levels raise `NotImplementedError`, because the old generic
formula could produce negative critical values. Prefer Monte Carlo simulation
of this exact implementation at the actual sample size.

## References

[Batsidis, Economou and Tzavelas (2016), Tests of fit for a lognormal distribution,
Journal of Statistical Computation and Simulation 86(2), 215–235](https://doi.org/10.1080/00949655.2014.1003138).
The publisher abstract verifies the KL size-bias symmetry motivation; full text
was inaccessible. The formulas above were checked independently with 70-digit
arithmetic and quadrature, not certified as an exact reproduction of every
normalization and integration domain in that paper.

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
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import KLIntegralLogNormalGoFStatistic

parameters = LogNormal.DEFAULT.parse({})
statistic = KLIntegralLogNormalGoFStatistic(parameters)
sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1])
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
