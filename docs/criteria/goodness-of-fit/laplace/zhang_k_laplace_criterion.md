# Zhang Z_K test for the Laplace distribution

## Description

`ZhangKLaplaceGofStatistic` tests the fully specified null $H_0: X \sim \operatorname{Laplace}(t,s)$.
Its algorithm code is `ZHANG_ZK_LAPLACE_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Laplace
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    ZhangKLaplaceGofStatistic,
)

parameters = Laplace.DEFAULT.parse({"t": 0, "s": 1})
criterion = ZhangKLaplaceGofStatistic(parameters)
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
Z_K=\max_{1\le i\le n}\left[(i-\tfrac12)\log\frac{(i-\tfrac12)/n}{u_i}+(n-i+\tfrac12)\log\frac{(n-i+\tfrac12)/n}{1-u_i}\right].
$$

This is the maximum likelihood-ratio EDF statistic in Zhang (2002), equation (3.1). Analytic Laplace log-CDF and log-survival values avoid infinities caused solely
by rounded CDF probabilities. Extreme statistics can still exceed float64 range. Logarithms are evaluated separately in the implementation to avoid overflowing probability ratios.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Zhang, J. (2002). [Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337). *Journal of the Royal Statistical Society: Series B*, 64(2), 281–294.

See the [Laplace criterion catalog](index.md) for the other implementations.

See [mathematical validation](mathematical-validation.md) for implementation checks and limits.
