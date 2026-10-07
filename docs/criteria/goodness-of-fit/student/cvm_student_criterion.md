# Cramer-von Mises W-squared for a specified Student t CDF.

## Hypothesis and formula

The null fixes df, loc and scale on the whole real line. Write
u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
W2 = 1/(12*n) + sum((u_i-(i-0.5)/n)**2).
No finite-sample correction is applied.
Large values reject. The probability integral transform removes all
distribution parameters from the null law, not from the hypothesis.
The reference describes a general statistic, applied here through the
specified Student CDF, not a separately derived Student-specific test.

## Parameters

```text
parameters : ParameterValues
    Values with df, loc, scale fixed; omitted parameters are unknown.
```

## Input and result

`execute_statistic(rvs, **kwargs)` accepts a finite, real, one-dimensional sample,
with at least one observation; ties and constants are allowed. It returns one scalar, leaves the input unchanged,
and ignores extra keyword arguments. Invalid samples and unrepresentable numerical
transformations raise `ValueError`. No preliminary fit or bootstrap is required.

## Example

```python
from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.student import CramerVonMisesStudentGofStatistic

statistic = CramerVonMisesStudentGofStatistic(StudentDistributionDescriptor.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1}))
value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
print(value)
```

## Scientific source

J. Zhang (2002), Powerful goodness-of-fit tests based on the likelihood
ratio, JRSS B 64, 281-294, https://doi.org/10.1111/1467-9868.00337. Section 2.3.

## Author(s)

Dmitriy Rusanov, Alexey Mironov
