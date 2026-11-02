# Scaled Kuiper test for uniformity

## Description

`KuiperScaledUniformGofStatistic` tests the fully specified null $H_0: X \sim U(a,b)$.
Its algorithm code is `KUIPER_SCALED_UNIFORM_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    KuiperScaledUniformGofStatistic,
)

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
criterion = KuiperScaledUniformGofStatistic(parameters)
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
V_{n,\mathrm{scaled}}=\sqrt n\,(D_n^++D_n^-).
$$

Here $D_n^+=\max_i(i/n-u_i)$ and $D_n^-=\max_i(u_i-(i-1)/n)$. The result is exactly $\sqrt n$ times the [unscaled Kuiper statistic](kuiper_uniform_criterion.md). At a fixed sample size the ordering is unchanged, but critical values must use the same scale.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Lemeshko, B. Yu., Gorbunova, A. A., Lemeshko, S. B., and Rogozhnikov, A. P. (2014). [Solving Problems of Using Some Nonparametric Goodness-of-Fit Tests](https://doi.org/10.3103/S875669901401004X). *Optoelectronics, Instrumentation and Data Processing*, 50, 21–35. The Kuiper section gives the scaled statistic and the correction in equation (3).

Kuiper, N. H. (1960). [Tests concerning random points on a circle](https://doi.org/10.1016/S1385-7258(60)50006-0). *Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A*, 63, 38–47.

Stephens, M. A. (1970). [Use of the Kolmogorov–Smirnov, Cramér–Von Mises and Related Statistics Without Extensive Tables](https://doi.org/10.1111/j.2517-6161.1970.tb00821.x). *Journal of the Royal Statistical Society: Series B*, 32(1), 115–122.

See the [uniformity criterion catalog](index.md) for the other implementations.
