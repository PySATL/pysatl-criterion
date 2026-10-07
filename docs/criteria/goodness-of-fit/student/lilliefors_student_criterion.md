# Fitted KS distance for Student t with known degrees of freedom.

## Hypothesis and formula

The null is t_df with unknown location and positive scale. Each call
estimates location by the sample median and scale by
(Q_0.75-Q_0.25)/(2*t_df.ppf(0.75)), using linear sample quantiles.
Return max(max(i/n-u_i), max(u_i-(i-1)/n)) for the fitted CDF.
This quantile estimator works without finite moments, including df <= 2;
it is not maximum likelihood. Samples need n >= 2 and positive IQR.
Ties are accepted if the IQR remains positive. Estimates are not stored.

This is a locally specified Lilliefors-type extension. A search did not
identify a primary paper establishing this exact Student/quantile version.
The normal Lilliefors paper does not validate it. Large values reject.
Location/scale equivariance removes those nuisance parameters; the null
law still depends on df, sample size and the estimator. Refit each
Monte Carlo replicate; ordinary KS and normal Lilliefors tables do not
apply. The built-in sampler requires complete Student parameters and
rejects this composite hypothesis. Calibrate externally at the fixed df
(location 0, scale 1 is justified by equivariance), calling this method
on every replicate. Previously stored LILLIE distributions are invalid.

## Parameters

```text
parameters : ParameterValues
    Values with df fixed; omitted parameters are unknown.
```

## Input and result

`execute_statistic(rvs, **kwargs)` accepts a finite, real, one-dimensional sample,
with at least two observations and positive interquartile range. It returns one scalar, leaves the input unchanged,
and ignores extra keyword arguments. Invalid samples and unrepresentable numerical
transformations raise `ValueError`. No preliminary fit or bootstrap is required.

## Example

```python
from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.student import LillieforsStudentGofStatistic

statistic = LillieforsStudentGofStatistic(StudentDistributionDescriptor.DEFAULT.parse({'df': 5}))
value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
print(value)
```

## Author(s)

Dmitriy Rusanov, Alexey Mironov
