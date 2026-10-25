# Liao–Shimokawa test for uniformity

## Description

`LiaoShimokawaUniformGofStatistic` tests the fully specified null $H_0: X \sim U(a,b)$.
Its algorithm code is `LIAO_SHIMOKAWA_UNIFORM_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    LiaoShimokawaUniformGofStatistic,
)

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
criterion = LiaoShimokawaUniformGofStatistic(parameters)
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
L_n = \frac{1}{\sqrt n}\sum_{i=1}^{n}\frac{\max\{i/n-u_i,\;u_i-(i-1)/n\}}{\sqrt{u_i(1-u_i)}}.
$$

The statistic divides each local EDF deviation by a tail-dependent weight. A CDF value of 0 or 1 produces positive infinity. The cited paper studies fitted extreme-value and Weibull models; this criterion uses a fully specified uniform CDF. Its calibration must therefore use the specified uniform null, not fitted-model tables from the paper.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Liao, M., and Shimokawa, T. (1999). [A new goodness-of-fit test for type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965). *Journal of Statistical Computation and Simulation*, 64(1), 23–48.

See the [uniformity criterion catalog](index.md) for the other implementations.
