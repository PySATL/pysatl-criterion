# Zhang K for Gamma

`ZhangKGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$. Define $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
Z_K=\max_i\left[r_i\log\frac{r_i}{nu_i}+(n-r_i)\log\frac{n-r_i}{n(1-u_i)}\right],\quad r_i=i-1/2
$$

Computed using logarithmic probabilities; zero observations give infinity.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Zhang (2002), §§2–3](https://doi.org/10.1111/1467-9868.00337).

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import ZhangKGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = ZhangKGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
