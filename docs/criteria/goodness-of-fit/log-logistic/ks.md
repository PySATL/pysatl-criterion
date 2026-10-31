# KolmogorovSmirnovLogLogisticGofStatistic

Kolmogorov–Smirnov for a fully specified log-logistic distribution.

[All log-logistic criteria](index.md) · [Mathematical verification](mathematical-validation.md)

## Hypothesis and input

Pass `Distribution.DEFAULT.parse({"alpha": alpha, "beta": beta})` as `parameters`.
Both scale `alpha` and shape `beta` must be fixed, finite and strictly positive.
Neither parameter is estimated. The sample must be a nonempty, finite, strictly
positive one-dimensional array; ties and a single observation are accepted.
Censored samples are not supported. The returned value is a scalar statistic;
large values reject the null (right tail).

## Definition

For ordered observations, $u_i=F_{\alpha,\beta}(x_{(i)})$,
$F_{\alpha,\beta}(x)=1/[1+(\alpha/x)^\beta]$ for $x>0$.
Write $r_i=i-1/2$, $D^+=\max_i(i/n-u_i)$ and
$D^-=\max_i(u_i-(i-1)/n)$.

$$
D=\max(D^+,D^-)
$$

The default returns unscaled $D$. The legacy keyword `alternative_type` can select `AlternativeType.RIGHT` ($D^+$) or `LEFT` ($D^-$); rejection is still in the right tail. Prefer the separate D+ and D− classes for distinct criterion identifiers.

## Calibration

For a fixed continuous null, EDF statistics and equal-probability cell counts
can be calibrated through independent Uniform(0,1) samples. Other binning rules
require the actual null and the same rule in every replicate. A rounded-data
null must reproduce rounding. Passing parameters estimated from this sample
as fixed values does not account for fitting.

The library currently has no built-in log-logistic random generator: use external
simulation (for example `scipy.stats.fisk`). This class computes a statistic,
not a p-value. For frequency statistics, the chi-square law with $k-1$ degrees
of freedom is an asymptotic approximation for fixed cells with adequate expected
counts, not an exact small-sample calibration. Binning is included in the
frequency criterion's storage configuration.

## Example

```python
from pysatl_criterion.distribution import LogLogisticDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic import (
    KolmogorovSmirnovLogLogisticGofStatistic,
)

parameters = Distribution.DEFAULT.parse({"alpha": 1.0, "beta": 1.0})
statistic = KolmogorovSmirnovLogLogisticGofStatistic(
    parameters
)
value = statistic.execute_statistic([0.1, 0.2, 0.5, 0.8, 1.5, 2.0, 4.0, 8.0])
assert value >= 0
```

## Scientific references

[Zhang (2002), §§2–3](https://doi.org/10.1111/1467-9868.00337) ([full text](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf))
