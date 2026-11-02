# Kolmogorov–Smirnov D− test for the Laplace distribution

## Description

`KolmogorovSmirnovDMinusLaplaceGofStatistic` tests the fully specified null $H_0: X \sim \operatorname{Laplace}(t,s)$.
Its algorithm code is `KS_MINUS_LAPLACE_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Laplace
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    KolmogorovSmirnovDMinusLaplaceGofStatistic,
)

parameters = Laplace.DEFAULT.parse({"t": 0, "s": 1})
criterion = KolmogorovSmirnovDMinusLaplaceGofStatistic(parameters)
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
D_n^- = \max_{1\le i\le n}\left(u_i-\frac{i-1}{n}\right).
$$

This is the unscaled negative EDF deviation. Large values indicate an empirical CDF below the null CDF, corresponding to observations tending toward larger values. The statistic uses a right-tail rejection region even though the EDF deviation is one-sided.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Smirnov, N. (1948). [Table for Estimating the Goodness of Fit of Empirical Distributions](https://doi.org/10.1214/aoms/1177730256). *Annals of Mathematical Statistics*, 19(2), 279–281.

See the [Laplace criterion catalog](index.md) for the other implementations.

See [mathematical validation](mathematical-validation.md) for implementation checks and limits.
