# Kuiper for exponentiated Weibull

`KuiperExponentiatedWeibullGofStatistic`

Short code: `KUIPER`. Full code: `KUIPER_EXPONENTIATED_WEIBULL_GOODNESS_OF_FIT`.

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
V=D^++D^-,\qquad D^+=\max_i(i/n-u_i),\quad D^-=\max_i(u_i-(i-1)/n).
$$

This is the unscaled Kuiper statistic. Do not interchange its critical values with those for the scaled or corrected classes.

## Calibration

Reject for large values (right tail). Use the same sample size and statistic normalization when calibrating the fixed null. The probability integral transform gives a Uniform(0,1) null in exact arithmetic. Critical values from tests that estimate Weibull parameters are unsuitable. The family currently rejects stored unversioned calibrations; regenerate them and do not assume storage loading is enabled.

## Example

```python
from pysatl_criterion.distribution import ExponentiatedWeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    KuiperExponentiatedWeibullGofStatistic,
)

parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
test = KuiperExponentiatedWeibullGofStatistic(parameters)
value = test.execute_statistic([0.1, 0.25, 0.4, 0.55, 0.7, 0.9, 1.2, 1.5, 2.0])
print(value)
```

## References

[Lemeshko et al. (2014), Solving Problems of Using Some Nonparametric Goodness-of-Fit Tests](https://doi.org/10.3103/S875669901401004X), eqs. (1)–(6); [full text](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf).

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
