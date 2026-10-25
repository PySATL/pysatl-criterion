# Anderson–Darling for Gamma

`AndersonDarlingGammaGofStatistic`

Use fixed positive shape `alfa` and **rate** `beta` (SciPy scale is `1/beta`), with origin zero. Both parameters must be supplied; no fitting occurs. Samples must be finite, real, nonempty, one-dimensional and nonnegative. Zero is accepted as a support boundary. Ties are allowed unless they make the statistic infinite or degenerate.

## Definition

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$. Define $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.

$$
A^2=-n-\frac1n\sum_{i=1}^n(2i-1)[\log u_i+\log(1-u_{n+1-i})]
$$

Evaluated with log-CDF and log-survival values. A zero observation gives infinity; no finite-sample correction is applied.

Large values oppose the null (right tail). `execute_statistic` returns a scalar, not a p-value. The sample is not modified.

## Source

[Zhang (2002), §§2–3](https://doi.org/10.1111/1467-9868.00337); [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437)..

## Usage

```python
from pysatl_criterion.distribution import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.gamma import AndersonDarlingGammaGofStatistic

parameters = GammaDistributionDescriptor.DEFAULT.parse({"alfa": 2.5, "beta": 1.5})
statistic = AndersonDarlingGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

See the [Gamma catalog](index.md) and [mathematical verification](mathematical-validation.md)
for calibration, numerical limits and the scope of scientific attribution.
