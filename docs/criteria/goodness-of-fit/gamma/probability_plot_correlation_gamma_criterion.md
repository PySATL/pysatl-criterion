# Probability plot correlation for Gamma

`ProbabilityPlotCorrelationGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$. Define $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
T=1-\operatorname{corr}(x_{(i)},q_i),\quad q_i=F_{\alpha,1}^{-1}\left(\frac{i-3/8}{n+1/4}\right)
$$

The citation supplies methodological background, not validation of this Gamma/Blom variant. The rate cancels from correlation. Positive affine transformations of the sample leave the statistic unchanged, so it cannot detect scale or location alternatives within the allowed support. Requires at least two nonconstant observations; at n=2 it is identically zero in exact arithmetic. Its null law depends on shape. Normality tables must not be reused.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Filliben (1975), The Probability Plot Correlation Coefficient Test for Normality](https://doi.org/10.1080/00401706.1975.10489279).

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import ProbabilityPlotCorrelationGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = ProbabilityPlotCorrelationGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
