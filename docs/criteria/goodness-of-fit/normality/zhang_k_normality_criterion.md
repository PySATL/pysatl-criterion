# Zhang K for normality

**Class:** `ZhangKNormalityGofStatistic`

**Code:** `ZHANG_ZK_NORMALITY_GOODNESS_OF_FIT`

## Hypothesis and input

H0 is the normal family with unknown mean and positive variance. Pass
`NormalDistributionDescriptor.DEFAULT.parse({})`. Both parameters are fitted on
every sample; the variance estimator uses `ddof=1`. At least two real, finite,
unmasked, one-dimensional observations with nonzero dispersion are required.
Ties are accepted. The sample is not modified and no fit is retained.
Large values reject H0. Calibration repeats fitting on each simulated sample.

## Formula

For ordered data, let `z_i=(x_(i)-mean(x))/s1`, `s1=std(x, ddof=1)`,
`u_i=Phi(z_i)`, `D+=max(i/n-u_i)`, and `D−=max(u_i-(i-1)/n)`.

`max_i [(i-1/2) log((i-1/2)/(n*u_i)) + (n-i+1/2) log((n-i+1/2)/(n*(1-u_i)))]`.

## Numerical details

Inputs are sorted without mutation. Logarithmic kernels use direct normal
log-CDF and log-SF; no clipping of tail probabilities is used. A fitted EDF
statistic needs its own simulation calibration. Knowing the algebraic formula
does not justify using fully specified-distribution tables.

## References

[Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337). The formula is an EDF functional; its application here refits normal parameters.

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import ZhangKNormalityGofStatistic
import numpy as np

statistic = ZhangKNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
assert statistic.hypothesis().parameters() == {}
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
