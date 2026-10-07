# Anderson-Darling A-squared for a specified Student t CDF.

## Hypothesis and formula

The null fixes df, loc and scale on the whole real line. Write
u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
A2 = -n - sum((2*i-1)*(log(u_i)+log(1-u_(n+1-i))))/n.
No fitted-parameter correction is applied.
Large values reject. The probability integral transform removes all
distribution parameters from the null law, not from the hypothesis.
The reference describes a general statistic, applied here through the
specified Student CDF, not a separately derived Student-specific test.
Log-CDF and log-survival are evaluated separately without clipping.
True probabilities are interior for finite data; numerical tail
underflow raises ValueError instead of fabricating a finite statistic.

## Parameters

df : float, default: 1
Fixed finite degrees of freedom, strictly positive.
loc : float, default: 0
Fixed finite location.
scale : float, default: 1
Fixed finite scale, strictly positive.

## Input and result

`execute_statistic(rvs, **kwargs)` accepts a finite, real, one-dimensional sample,
with at least one observation; ties and constants are allowed. It returns one scalar, leaves the input unchanged,
and ignores extra keyword arguments. Invalid samples and unrepresentable numerical
transformations raise `ValueError`. No preliminary fit or bootstrap is required.

## Example

```python
from pysatl_criterion.statistics.goodness_of_fit.student import AndersonDarlingStudentGofStatistic

statistic = AndersonDarlingStudentGofStatistic(df=5)
value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
print(value)
```

## Scientific source

J. Zhang (2002), Powerful goodness-of-fit tests based on the likelihood
ratio, JRSS B 64, 281-294, https://doi.org/10.1111/1467-9868.00337. Section 2.2.

## Author(s)

Dmitriy Rusanov, Alexey Mironov
