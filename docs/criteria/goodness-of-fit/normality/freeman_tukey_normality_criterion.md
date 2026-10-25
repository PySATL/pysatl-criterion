# Freeman–Tukey for normality

**Class:** `FreemanTukeyNormalityGofStatistic`

**Code:** `FREEMAN_TUKEY_NORMALITY_GOODNESS_OF_FIT`

## Hypothesis and input

H0 is the normal family with unknown mean and positive variance. Pass
`NormalDistributionDescriptor.DEFAULT.parse({})`. Both parameters are fitted on
every sample; the variance estimator uses `ddof=1`. At least two real, finite,
unmasked, one-dimensional observations with nonzero dispersion are required.
Ties are accepted. The sample is not modified and no fit is retained.
Large values reject H0. Calibration repeats fitting on each simulated sample.

## Formula

Let `O_j` be the observed count and `E_j=n*p_j>0` the fitted expected count.

`4 Σ (sqrt(O_j)-sqrt(E_j))²`. A zero observed count contributes `4*E_j`.

## Binning and calibration

The optional `binning` keyword accepts a `BinningStrategy`; the default is
`EqualProbabilityBins()`. `ExplicitBins` boundaries are in the original data units.
Fit mean and unbiased variance on each replicate, then rebuild the same bins.
Do not assume a universal chi-square distribution with `k-3` degrees of freedom:
raw-data estimates, adaptive boundaries and the bin-count rule affect calibration.
Zero-probability categories and any configured minimum-expected-count merging
follow the [shared binning contract](../binning/index.md).

## References

[Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x).

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import FreemanTukeyNormalityGofStatistic
import numpy as np

statistic = FreemanTukeyNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
assert statistic.hypothesis().parameters() == {}
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
