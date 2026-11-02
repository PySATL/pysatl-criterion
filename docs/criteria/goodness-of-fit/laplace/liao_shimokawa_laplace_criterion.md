# Liao–Shimokawa test for the Laplace distribution

## Description

`LiaoShimokawaLaplaceGofStatistic` tests the fully specified null $H_0: X \sim \operatorname{Laplace}(t,s)$.
Its algorithm code is `LIAO_SHIMOKAWA_LAPLACE_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Laplace
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    LiaoShimokawaLaplaceGofStatistic,
)

parameters = Laplace.DEFAULT.parse({"t": 0, "s": 1})
criterion = LiaoShimokawaLaplaceGofStatistic(parameters)
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
L_n = \frac{1}{\sqrt n}\sum_{i=1}^{n}\frac{\max\{i/n-u_i,\;u_i-(i-1)/n\}}{\sqrt{u_i(1-u_i)}}.
$$

The statistic divides each local EDF deviation by a tail-dependent weight. Analytic Laplace log-CDF and log-survival values avoid infinities caused solely
by rounded CDF probabilities. Extreme statistics can still exceed float64 range. The cited paper studies fitted extreme-value and Weibull models; this criterion uses a fully specified Laplace CDF. Its calibration must therefore use the specified Laplace null, not fitted-model tables from the paper.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Liao, M., and Shimokawa, T. (1999). [A new goodness-of-fit test for type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965). *Journal of Statistical Computation and Simulation*, 64(1), 23–48.

See the [Laplace criterion catalog](index.md) for the other implementations.

See [mathematical validation](mathematical-validation.md) for implementation checks and limits.
