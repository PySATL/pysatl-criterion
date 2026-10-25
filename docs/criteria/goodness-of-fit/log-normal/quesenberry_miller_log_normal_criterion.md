# Quesenberry-Miller test for log-normal distribution

**Class:** `QuesenberryMillerLogNormalGofStatistic`

**Code:** `QUESENBERRY_MILLER_LOGNORMAL_GOODNESS_OF_FIT`

## Hypothesis and inputs

Both parameters must be supplied before observing the sample: `mu` is the
mean of log(X), `s > 0` is its standard deviation. The alternative coordinates
are `s` and `scale = exp(mu)`, the **median**, not the mean of X.
Use `LOG_LOCATION_SCALE.parse({"mu": ..., "s": ...})` or
`SHAPE_SCALE.parse({"s": ..., "scale": ...})`. Partial and empty hypotheses
are rejected. No parameters are fitted to the sample.

The sample must be nonempty, finite, real and one-dimensional; masked values
are rejected. Ties are permitted. EDF/spacing formulas use the null CDF, which
is zero at nonpositive observations; finite penalties are possible. This is
not the same policy as logarithmic family statistics, which reject such input.

## Statistic

Set $u_i=F_{\mu,s}(x_{(i)})$, $u_0=0$, $u_{n+1}=1$ and
$D_i=u_i-u_{i-1}$ for $i=1,\ldots,n+1$. Then

$$Q=\sum_{i=1}^{n+1}D_i^2+\sum_{i=1}^{n}D_iD_{i+1}.$$

There is no cyclic term $D_{n+1}D_1$. All n+1 spacings, including both
boundary gaps, are present. Uniform quantile samples provide an exact
hand-calculated reference. The null CDF must be specified in advance.

## References

- [Quesenberry and Miller (1977), Power studies of some tests for uniformity](https://doi.org/10.1080/00949657708810150).
- The explicit noncyclic formula is also given in the authors' comparative study,
  [Sreedevi and Kattumannil, §4.1](https://arxiv.org/html/2106.06368v1#S4.SS1).

## Rejection and calibration

Rejection tail: **right**. A statistic value alone is not a p-value.
Use the fixed null and the same sample size, settings and binning for calibration.
Do not substitute fitted-normality tables.

## Examples

```python
import numpy as np
from pysatl_criterion.distribution import LogNormalDistributionDescriptor as LogNormal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import QuesenberryMillerLogNormalGofStatistic

parameters = LogNormal.DEFAULT.parse({'mu': 0.4, 's': 0.8})
statistic = QuesenberryMillerLogNormalGofStatistic(parameters)
sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1])
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
