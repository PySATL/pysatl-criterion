# Freeman–Tukey test for the Laplace distribution

## Description

`FreemanTukeyLaplaceGofStatistic` tests the fully specified null $H_0: X \sim \operatorname{Laplace}(t,s)$.
Its algorithm code is `FREEMAN_TUKEY_LAPLACE_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Laplace
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    FreemanTukeyLaplaceGofStatistic,
)

parameters = Laplace.DEFAULT.parse({"t": 0, "s": 1})
criterion = FreemanTukeyLaplaceGofStatistic(parameters, binning=EqualProbabilityBins(count=3))
statistic = criterion.execute_statistic([-2.0, -1.0, -0.2, 0.2, 1.0, 2.0])
print(statistic)
```

## Arguments

`parameters` — `ParameterValues` with finite location `t` and positive finite
scale `s`, both fixed before observing the sample. Missing parameters are
unsupported. No parameters are estimated by this criterion.

`rvs` — a nonempty, one-dimensional sample of finite real observations.

`binning` — keyword-only strategy, default `EqualProbabilityBins()`:
`PowerCount(coefficient=2.0, exponent=0.4)`, `min_expected=None`, `max_bins=1000`.
The support is the entire real line. Ties and constant samples are accepted.
Bins must cover the full support; expected counts are obtained from the fixed
Laplace null. No minimum expected count is imposed by default.

## Details

The default equal-probability bins are built in Laplace probability space,
not as equal-width intervals on the real line. With $m$ such bins,
$p_j=1/m$. Other strategies use their actual null bin probabilities.
Unknown or sample-fitted parameters are not supported. A fitted extension would
need refitting and rebuilding bins in every Monte Carlo replicate.


For observed bin counts $O_j$ and positive expected counts $E_j=np_j$,
where $p_j$ is the bin probability under the specified Laplace null:

$$
T_{FT}=4\sum_{j=1}^{m}\left(\sqrt{O_j}-\sqrt{E_j}\right)^2.
$$

The power-divergence parameter is $\lambda=-1/2$. An empty observed bin contributes $4E_j$, so zero observed counts do not make this statistic infinite.

Large values form the right-tail rejection region. Binning is reapplied for every sample. Calibrate with the same rule and sample
size; a statistic value alone is not a p-value. The default has no minimum
expected-count constraint, so a chi-squared approximation is not automatically
justified. See [binning strategies](../binning/index.md) and
[calibration storage](../binning/calibration.md). `configuration()` includes the
binning policy, while `code()` identifies the algorithm.

## References

Cressie, N., and Read, T. R. C. (1984). [Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x). *Journal of the Royal Statistical Society: Series B*, 46(3), 440–464.

See the [Laplace criterion catalog](index.md) for the other implementations.

See [mathematical validation](mathematical-validation.md) for implementation checks and limits.
