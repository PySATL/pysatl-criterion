# Kolmogorov–Smirnov for exponentiated Weibull

`KolmogorovSmirnovExponentiatedWeibullGofStatistic`

Short code: `KS`. Full code: `KS_EXPONENTIATED_WEIBULL_GOODNESS_OF_FIT`.

## Hypothesis and input

The null is fully specified:

$$
F_{a,k,s}(x)=\left[1-\exp\left(-(x/s)^k\right)\right]^a,\quad x\ge0,
\qquad a,k,s>0.
$$

Fix `exponent=a`, `shape=k`, `scale=s` in
`ExponentiatedWeibullDistributionDescriptor.DEFAULT.parse(...)`; location is zero.
All three parameters are required and are never estimated from the observations.
`execute_statistic` accepts a nonempty one-dimensional finite nonnegative sample;
ties and constant samples are allowed. It returns a scalar statistic, not a p-value,
and leaves the sample and parameters unchanged.

## Formula

Write $u_i=F_{a,k,s}(x_{(i)})$ for the sorted observations, $i=1,\ldots,n$.

$$
D=\max(D^+,D^-),\qquad D^+=\max_i(i/n-u_i),\quad D^-=\max_i(u_i-(i-1)/n).
$$

Returns the unsquared, unscaled D. The optional `alternative_type` accepts `AlternativeType.TWO_TAILED` (D), `RIGHT` (D+) or `LEFT` (D−). These are CDF directions; all three reject for large values. `mode` accepts `auto`, `exact`, `asymp`, `approx` for compatibility and does not change the statistic.

## Calibration

Reject for large values (right tail). Use the same sample size and statistic normalization when calibrating the fixed null. The probability integral transform gives a Uniform(0,1) null in exact arithmetic. Critical values from tests that estimate Weibull parameters are unsuitable. The family currently rejects stored unversioned calibrations; regenerate them and do not assume storage loading is enabled.

## Example

```python
from pysatl_criterion.distribution import ExponentiatedWeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    KolmogorovSmirnovExponentiatedWeibullGofStatistic,
)

parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
test = KolmogorovSmirnovExponentiatedWeibullGofStatistic(parameters)
value = test.execute_statistic([0.1, 0.25, 0.4, 0.55, 0.7, 0.9, 1.2, 1.5, 2.0])
print(value)
```

## References

[Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §§2–3; [full text](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf).

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
