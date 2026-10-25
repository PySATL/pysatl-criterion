# Corrected Kuiper for Gamma

`KuiperCorrectedGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$. Define $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
V_c=(\sqrt n+0.155+0.24/\sqrt n)(D^++D^-)
$$

This is a finite-sample scaling, not an exact finite-sample p-value.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Lemeshko et al. (2014), eqs. (1)–(6)](https://doi.org/10.3103/S875669901401004X).

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import KuiperCorrectedGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = KuiperCorrectedGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
