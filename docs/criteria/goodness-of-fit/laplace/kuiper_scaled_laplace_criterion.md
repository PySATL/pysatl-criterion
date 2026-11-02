# Scaled Kuiper test for the Laplace distribution

## Description

`KuiperScaledLaplaceGofStatistic` tests the fully specified null $H_0: X \sim \operatorname{Laplace}(t,s)$.
Its algorithm code is `KUIPER_SCALED_LAPLACE_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Laplace
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    KuiperScaledLaplaceGofStatistic,
)

parameters = Laplace.DEFAULT.parse({"t": 0, "s": 1})
criterion = KuiperScaledLaplaceGofStatistic(parameters)
statistic = criterion.execute_statistic([-2.0, -1.0, -0.2, 0.2, 1.0, 2.0])
print(statistic)
```

## Arguments

`parameters` — `ParameterValues` with finite location `t` and positive finite
scale `s`, both fixed before observing the sample. Missing parameters are
unsupported. No parameters are estimated by this criterion.

`rvs` — a nonempty, one-dimensional sample of finite real observations.

The support is the entire real line. Singletons, ties and constant samples are
accepted. Empty, nonfinite, complex and multidimensional samples are rejected.

## Details

Writing $z=(x-t)/s$, the Laplace CDF is $F_{t,s}(x)=e^z/2$ for $z<0$
and $F_{t,s}(x)=1-e^{-z}/2$ otherwise. For a fixed continuous null the
probability integral transform gives independent uniforms. Its null law depends
on sample size, not the known location or scale. Parameters estimated from the
sample require a different test and calibration with refitting in every replicate.
Rounding-induced ties require a matching observation model during calibration.


Let $u_i=F_{t,s}(X_{(i)})$ for the sorted sample, with $n$ observations.

$$
V_{n,\mathrm{scaled}}=\sqrt n\,(D_n^++D_n^-).
$$

Here $D_n^+=\max_i(i/n-u_i)$ and $D_n^-=\max_i(u_i-(i-1)/n)$. The result is exactly $\sqrt n$ times the [unscaled Kuiper statistic](kuiper_laplace_criterion.md). At a fixed sample size the ordering is unchanged, but critical values must use the same scale.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Lemeshko, B. Yu., Gorbunova, A. A., Lemeshko, S. B., and Rogozhnikov, A. P. (2014). [Solving Problems of Using Some Nonparametric Goodness-of-Fit Tests](https://doi.org/10.3103/S875669901401004X). *Optoelectronics, Instrumentation and Data Processing*, 50, 21–35. The Kuiper section gives the scaled statistic and the correction in equation (3).

Kuiper, N. H. (1960). [Tests concerning random points on a circle](https://doi.org/10.1016/S1385-7258(60)50006-0). *Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A*, 63, 38–47.

Stephens, M. A. (1970). [Use of the Kolmogorov–Smirnov, Cramér–Von Mises and Related Statistics Without Extensive Tables](https://doi.org/10.1111/j.2517-6161.1970.tb00821.x). *Journal of the Royal Statistical Society: Series B*, 32(1), 115–122.

See the [Laplace criterion catalog](index.md) for the other implementations.

See [mathematical validation](mathematical-validation.md) for implementation checks and limits.
