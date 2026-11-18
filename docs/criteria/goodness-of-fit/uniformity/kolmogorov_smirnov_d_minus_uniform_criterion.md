# Kolmogorov–Smirnov D− test for uniformity

## Description

`KolmogorovSmirnovDMinusUniformGofStatistic` tests the fully specified null $H_0: X \sim U(a,b)$.
Its algorithm code is `KS_MINUS_UNIFORM_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    KolmogorovSmirnovDMinusUniformGofStatistic,
)

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
criterion = KolmogorovSmirnovDMinusUniformGofStatistic(parameters)
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
D_n^- = \max_{1\le i\le n}\left(u_i-\frac{i-1}{n}\right).
$$

This is the unscaled negative EDF deviation. Large values indicate an empirical CDF below the null CDF, corresponding to observations tending toward larger values. The statistic uses a right-tail rejection region even though the EDF deviation is one-sided.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Smirnov, N. (1948). [Table for Estimating the Goodness of Fit of Empirical Distributions](https://doi.org/10.1214/aoms/1177730256). *Annals of Mathematical Statistics*, 19(2), 279–281.

See the [uniformity criterion catalog](index.md) for the other implementations.
