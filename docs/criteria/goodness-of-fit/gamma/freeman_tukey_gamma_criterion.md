# Freeman–Tukey for Gamma

`FreemanTukeyGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Here $O_j$ and $E_j=np_j$ are observed and expected cell counts. With k equal-probability cells, $E_j=n/k$.

$$
T=4\sum_j(\sqrt{O_j}-\sqrt{E_j})^2
$$

An empty observed cell contributes $4E_j$.

The constructor accepts `binning`, a `BinningStrategy`. The default is `EqualProbabilityBins()` with `PowerCount()`: k is ceil(2*n**0.4), bounded to 2…1000. To fix k, use `EqualProbabilityBins(count=k)`. Counts use the intervals of the transformed CDF; the last interval includes 1. Other strategies use their own null probabilities. Binning configuration is included in storage keys.

For fixed null-probability bins, use multinomial/Monte Carlo calibration with the same n and bins. The chi-square(k−1) approximation needs sufficiently large expected counts and fixed k; sparse cells or sample-dependent rules require separate calibration.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x).

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import FreemanTukeyGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = FreemanTukeyGammaGofStatistic(parameters, binning=EqualProbabilityBins(count=3))
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
