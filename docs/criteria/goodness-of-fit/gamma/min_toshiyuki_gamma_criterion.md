# Min–Toshiyuki (legacy name) for Gamma

`MinToshiyukiGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$. Define $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
L_n=\frac1{\sqrt n}\sum_{i=1}^n\frac{\max(i/n-u_i,u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}
$$

This is mathematically the same discrepancy as Liao–Shimokawa. The public name is retained for compatibility, not presented as a separate published Gamma test. Logarithmic tail probabilities are used. A zero observation gives infinity. The original fitted Weibull/extreme-value critical values do not apply to the fixed Gamma null.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965).

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import MinToshiyukiGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = MinToshiyukiGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
