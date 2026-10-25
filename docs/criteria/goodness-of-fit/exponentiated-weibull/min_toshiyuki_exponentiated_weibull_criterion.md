# Liao–Shimokawa (historical Min–Toshiyuki) for exponentiated Weibull

`MinToshiyukiExponentiatedWeibullGofStatistic`

Short code: `MT`. Full code: `MT_EXPONENTIATED_WEIBULL_GOODNESS_OF_FIT`.

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
L_n=\frac1{\sqrt n}\sum_{i=1}^n\frac{\max(i/n-u_i,\ u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}.
$$

The public name uses the given names of Min Liao and Toshiyuki Shimokawa. This is the weighted EDF expression applied to a **fixed** CDF. The original paper estimates parameters; its fitted-Weibull critical values do not apply here. The original full text was unavailable in this audit; eq. (13) in Al-Zahrani provides the formula cross-check, not calibration for this family.

An exact zero observation gives positive infinity. Finite tail penalties are retained using log probabilities when the CDF rounds to one; the statistic may still overflow when its true magnitude exceeds float64.

## Calibration

Reject for large values (right tail). Use the same sample size and statistic normalization when calibrating the fixed null. The probability integral transform gives a Uniform(0,1) null in exact arithmetic. Critical values from tests that estimate Weibull parameters are unsuitable. The family currently rejects stored unversioned calibrations; regenerate them and do not assume storage loading is enabled.

## Example

```python
from pysatl_criterion.distribution import ExponentiatedWeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponentiated_weibull import (
    MinToshiyukiExponentiatedWeibullGofStatistic,
)

parameters = Distribution.DEFAULT.parse({"exponent": 1, "shape": 2, "scale": 1})
test = MinToshiyukiExponentiatedWeibullGofStatistic(parameters)
value = test.execute_statistic([0.1, 0.25, 0.4, 0.55, 0.7, 0.9, 1.2, 1.5, 2.0])
print(value)
```

## References

[Liao and Shimokawa (1999), A new goodness-of-fit test for type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965). Formula cross-check: [Al-Zahrani (2012), Goodness-of-fit for the Topp-Leone distribution, eq. (13)](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf).

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
