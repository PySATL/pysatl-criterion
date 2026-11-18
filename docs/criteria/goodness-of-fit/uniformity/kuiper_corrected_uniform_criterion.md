# Corrected Kuiper test for uniformity

## Description

`KuiperCorrectedUniformGofStatistic` tests the fully specified null $H_0: X \sim U(a,b)$.
Its algorithm code is `KUIPER_CORRECTED_UNIFORM_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    KuiperCorrectedUniformGofStatistic,
)

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
criterion = KuiperCorrectedUniformGofStatistic(parameters)
statistic = criterion.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic)
```

## Arguments

`parameters` — `ParameterValues` with both `a` and `b` explicitly fixed and
`a < b`. Bounds are not estimated from the sample; omitted bounds are unsupported.

`rvs` — a nonempty, one-dimensional sample of finite real observations.

Values outside `[a, b]` are mapped by the null CDF to 0 or 1.

## Details

Let $u_i=F_{a,b}(X_{(i)})$ for the sorted sample, with $n$ observations.

$$
V_{n,\mathrm{corrected}}=\left(\sqrt n+0.155+\frac{0.24}{\sqrt n}\right)(D_n^++D_n^-).
$$

Here $D_n^+=\max_i(i/n-u_i)$ and $D_n^-=\max_i(u_i-(i-1)/n)$. This class returns the finite-sample corrected statistic. It does not calculate a p-value. The unscaled, scaled and corrected variants have distinct algorithm codes and require matching critical values.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Lemeshko, B. Yu., Gorbunova, A. A., Lemeshko, S. B., and Rogozhnikov, A. P. (2014). [Solving Problems of Using Some Nonparametric Goodness-of-Fit Tests](https://doi.org/10.3103/S875669901401004X). *Optoelectronics, Instrumentation and Data Processing*, 50, 21–35. The Kuiper section gives the scaled statistic and the correction in equation (3).

Stephens, M. A. (1970). [Use of the Kolmogorov–Smirnov, Cramér–Von Mises and Related Statistics Without Extensive Tables](https://doi.org/10.1111/j.2517-6161.1970.tb00821.x). *Journal of the Royal Statistical Society: Series B*, 32(1), 115–122.

Kuiper, N. H. (1960). [Tests concerning random points on a circle](https://doi.org/10.1016/S1385-7258(60)50006-0). *Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A*, 63, 38–47.

See the [uniformity criterion catalog](index.md) for the other implementations.
