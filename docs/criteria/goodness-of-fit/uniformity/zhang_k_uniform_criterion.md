# Zhang Z_K test for uniformity

## Description

`ZhangKUniformGofStatistic` tests the fully specified null $H_0: X \sim U(a,b)$.
Its algorithm code is `ZHANG_ZK_UNIFORM_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    ZhangKUniformGofStatistic,
)

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
criterion = ZhangKUniformGofStatistic(parameters)
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
Z_K=\max_{1\le i\le n}\left[(i-\tfrac12)\log\frac{(i-\tfrac12)/n}{u_i}+(n-i+\tfrac12)\log\frac{(n-i+\tfrac12)/n}{1-u_i}\right].
$$

This is the maximum likelihood-ratio EDF statistic in Zhang (2002), equation (3.1). A CDF value of 0 or 1 produces positive infinity. Logarithms are evaluated separately in the implementation to avoid overflowing probability ratios.

Large statistic values form the rejection region. `execute_statistic` returns
the statistic only. Use a null distribution calibrated for this class and sample
size, for example with `MonteCarloLimitDistributionResolver`. Parameters remain
fixed throughout calibration.

## References

Zhang, J. (2002). [Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337). *Journal of the Royal Statistical Society: Series B*, 64(2), 281–294.

See the [uniformity criterion catalog](index.md) for the other implementations.
