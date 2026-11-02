# Cressie–Read test for uniformity

## Description

`CressieReadUniformGofStatistic` tests the fully specified null $H_0: X \sim U(a,b)$.
Its algorithm code is `CRESSIE_READ_UNIFORM_GOODNESS_OF_FIT`.

## Usage

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.uniform import (
    CressieReadUniformGofStatistic,
)

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
criterion = CressieReadUniformGofStatistic(parameters, binning=EqualProbabilityBins(count=3))
statistic = criterion.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic)
```

## Arguments

`parameters` — `ParameterValues` with both `a` and `b` explicitly fixed and
`a < b`. Bounds are not estimated from the sample; omitted bounds are unsupported.

`rvs` — a nonempty, one-dimensional sample of finite real observations.

`binning` — keyword-only strategy, default `EqualProbabilityBins()`:
`PowerCount(coefficient=2.0, exponent=0.4)`, `min_expected=None`, `max_bins=1000`.
Observations must belong to `[a, b]`.

## Details

For observed bin counts $O_j$ and positive expected counts $E_j=np_j$,
where $p_j$ is the bin probability under the specified uniform null:

$$
T_{CR}=\frac95\sum_{j=1}^{m}O_j\left[\left(\frac{O_j}{E_j}\right)^{2/3}-1\right].
$$

This class fixes the power-divergence parameter at $\lambda=2/3$; there is no `lambda_` constructor argument. Terms with zero observed counts contribute zero by continuity.

Large values form the right-tail rejection region. Binning is reapplied for every sample. Calibrate with the same rule and sample
size; a statistic value alone is not a p-value. The default has no minimum
expected-count constraint, so a chi-squared approximation is not automatically
justified. See [binning strategies](../binning/index.md) and
[calibration storage](../binning/calibration.md). `configuration()` includes the
binning policy, while `code()` identifies the algorithm.

## References

Cressie, N., and Read, T. R. C. (1984). [Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x). *Journal of the Royal Statistical Society: Series B*, 46(3), 440–464.

See the [uniformity criterion catalog](index.md) for the other implementations.
