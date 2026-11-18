# Pearson chi-squared for log-normal data

**Class:** `Chi2PearsonLogNormalGofStatistic`

**Code:** `CHI2_PEARSON_LOGNORMAL_GOODNESS_OF_FIT`

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

## Statistic and binning

Pearson chi-square: sum((observed - expected)**2 / expected).

## References
Pearson, K. (1900). On the criterion that a given system of deviations
from the probable in the case of a correlated system of variables is
such that it can be reasonably supposed to have arisen from random sampling.
Philosophical Magazine, 50(302), 157-175.
https://doi.org/10.1080/14786440009463897

The constructor accepts `binning: BinningStrategy`; its default is
`EqualProbabilityBins()`. Bins use the fixed log-normal distribution in the
original sample units. Define observed counts O_j and expected counts
E_j = n * P(X in cell j). The zero-count limits are part of the definition;
Neyman and modified log-likelihood can return positive infinity when O_j=0<E_j.
A positive observed count in a zero-probability cell gives infinity.
For a fixed partition with k positive-probability cells, the usual multinomial
asymptotic reference has k-1 degrees of freedom. Adaptive bins require calibration
of the entire bin-building procedure; no automatic k-3 correction is justified.

## Rejection and calibration

Rejection tail: **right**. A statistic value alone is not a p-value.
Use the fixed null and the same sample size, settings and binning for calibration.
Do not substitute fitted-normality tables.

## Examples

```python
import numpy as np
from pysatl_criterion.distribution import LogNormalDistributionDescriptor as LogNormal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import (
    Chi2PearsonLogNormalGofStatistic,
)

parameters = LogNormal.DEFAULT.parse({"mu": 0.4, "s": 0.8})
statistic = Chi2PearsonLogNormalGofStatistic(parameters)
sample = np.exp([-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1])
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
