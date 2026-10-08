# Lilliefors test for exponentiality (1969)

## Description

The null hypothesis is an exponential distribution with unknown positive mean
and known origin zero. Estimate the mean by the sample mean on each call.
For sorted observations, define

$$
F_i=1-\exp(-x_{(i)}/\bar x),\qquad
D=\max_{1\le i\le n}\left\{\frac{i}{n}-F_i,\;F_i-\frac{i-1}{n}\right\}.
$$

Large values reject the null. The implementation returns this unscaled
statistic, without a p-value or a test decision.

## Usage

```python
from pysatl_criterion.distribution.distributions import ExponentialDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import LillieforsExponentialityGofStatistic

parameters = ExponentialDistributionDescriptor.DEFAULT.parse({})
statistic = LillieforsExponentialityGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
print(value)
```

`parameters` must leave the rate unknown. `rvs` must be a one-dimensional
sample of at least two finite, real, nonnegative observations with positive
mean. Zero observations and positive constant samples are accepted. Negative,
masked, nonfinite and all-zero samples raise `ValueError`. Extra keyword
arguments raise `TypeError`.

The short code is `LILLIE`, the full code is
`LILLIE_EXPONENTIALITY_GOODNESS_OF_FIT`, and the rejection tail is right.
The input is not modified and fitted parameters are not retained.

## Calibration

Dividing by the estimated mean makes the statistic invariant under positive
rescaling. Monte Carlo calibration can use samples from Exp(1), provided
`execute_statistic` is called on every replicate to estimate its own mean.
Use critical values for this fitted exponential statistic, not the ordinary
fixed-CDF Kolmogorov–Smirnov tables or the normal Lilliefors tables.

For a specified rate, use `KolmogorovSmirnovExponentialityGofStatistic` instead.

## Reference

H. W. Lilliefors (1969),
[On the Kolmogorov-Smirnov Test for the Exponential Distribution with Mean Unknown](https://doi.org/10.1080/01621459.1969.10500983),
Journal of the American Statistical Association 64(325), 387–389.
