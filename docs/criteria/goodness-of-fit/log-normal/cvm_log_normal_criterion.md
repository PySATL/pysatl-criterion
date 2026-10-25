# Cramer-von Mises test for log-normal data

**Class:** `CramerVonMiseLogNormalGofStatistic`

**Code:** `CVM_LOGNORMAL_GOODNESS_OF_FIT`

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

Set $u_i=\Phi((\log x_{(i)}-\mu)/s)$ for positive observations.
The following general EDF definition is evaluated at these fixed-null probabilities.

Cramér-von Mises W² statistic.

Computes 1/(12*n)+sum((u_i-(i-0.5)/n)**2).

## References
Zhang, J. (2002). Powerful goodness-of-fit tests based on the likelihood
ratio. JRSS B, 64(2), 281-294, Section 2.3.
https://doi.org/10.1111/1467-9868.00337

## Rejection and calibration

Rejection tail: **right**. A statistic value alone is not a p-value.
Use the fixed null and the same sample size, settings and binning for calibration.
Do not substitute fitted-normality tables.

## Examples

```python
import numpy as np
from pysatl_criterion.distribution import LogNormalDistributionDescriptor as LogNormal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import CramerVonMiseLogNormalGofStatistic

parameters = LogNormal.DEFAULT.parse({'mu': 0.4, 's': 0.8})
statistic = CramerVonMiseLogNormalGofStatistic(parameters)
sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1])
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
