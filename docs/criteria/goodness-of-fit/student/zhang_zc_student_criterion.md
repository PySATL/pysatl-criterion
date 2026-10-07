# Zhang Z_C for a specified Student t CDF.

## Hypothesis and formula

The null fixes df, loc and scale on the whole real line. Write
u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
Z_C = sum((log((1-u_i)/u_i) - log((n-i+0.25)/(i-0.75)))**2).
This implements equation (3.3), not the exact integrated likelihood
ratio from which Zhang derives that formula approximately.
Large values reject. The probability integral transform removes all
distribution parameters from the null law, not from the hypothesis.
The reference describes a general statistic, applied here through the
specified Student CDF, not a separately derived Student-specific test.
Log-CDF and log-survival are evaluated separately without clipping.
True probabilities are interior for finite data; numerical tail
underflow raises ValueError instead of fabricating a finite statistic.
Recompute stored null distributions made with the previous clipped code.

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
from pysatl_criterion.statistics.goodness_of_fit.student import ZhangZcStudentGofStatistic

statistic = ZhangZcStudentGofStatistic(StudentDistributionDescriptor.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1}))
value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
print(value)
```

## Scientific source

J. Zhang (2002), Powerful goodness-of-fit tests based on the likelihood
ratio, JRSS B 64, 281-294, https://doi.org/10.1111/1467-9868.00337. Section 3.3.

## Author(s)

Dmitriy Rusanov, Alexey Mironov
