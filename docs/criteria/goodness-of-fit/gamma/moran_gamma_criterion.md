# Moran for Gamma

`MoranGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$. Define $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
M=-\sum_{j=1}^{n+1}\log(nD_j),\quad D_j=u_j-u_{j-1},\ u_0=0,\ u_{n+1}=1
$$

The historical factor n is retained. Relative to scaling every spacing by n+1, the returned value is larger by $(n+1)\log((n+1)/n)$. Calibrate this normalization. Ties and zero observations give infinity. Log-CDF/log-survival differences preserve small spacings until the underlying probabilities themselves become unrepresentable.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Moran (1951), The Random Division of an Interval—Part II](https://doi.org/10.1111/j.2517-6161.1951.tb00079.x).

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import MoranGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = MoranGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
