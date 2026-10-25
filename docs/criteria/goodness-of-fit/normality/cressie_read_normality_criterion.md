# Cressie–Read for normality

**Class:** `CressieReadNormalityGofStatistic`

**Code:** `CRESSIE_READ_NORMALITY_GOODNESS_OF_FIT`

## Hypothesis and input

H0 is the normal family with unknown mean and positive variance. Pass
`NormalDistributionDescriptor.DEFAULT.parse({})`. Both parameters are fitted on
every sample; the variance estimator uses `ddof=1`. At least two real, finite,
unmasked, one-dimensional observations with nonzero dispersion are required.
Ties are accepted. The sample is not modified and no fit is retained.
Large values reject H0. Calibration repeats fitting on each simulated sample.

## Formula

Let `O_j` be the observed count and `E_j=n*p_j>0` the fitted expected count.

`(9/5) Σ O_j [(O_j/E_j)^(2/3)-1]`. Terms with zero observed count are zero by continuity.

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
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import CressieReadNormalityGofStatistic
import numpy as np

statistic = CressieReadNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
assert statistic.hypothesis().parameters() == {}
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
