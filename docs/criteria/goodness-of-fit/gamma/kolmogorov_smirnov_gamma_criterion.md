# Kolmogorov–Smirnov for Gamma

`KolmogorovSmirnovGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$. Define $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
D=\max(D^+,D^-)
$$

The default is the two-sided CDF discrepancy. Legacy `alternative_type=RIGHT` selects D+ and `LEFT` selects D−; each statistic rejects for large values. `mode` does not change the statistic. Nondefault directions cannot use stored calibration.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Zhang (2002), §§2–3](https://doi.org/10.1111/1467-9868.00337).

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import KolmogorovSmirnovGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = KolmogorovSmirnovGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
