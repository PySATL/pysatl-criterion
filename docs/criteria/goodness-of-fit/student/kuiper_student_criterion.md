# Kuiper V for a specified Student t CDF.

## Hypothesis and formula

The null fixes df, loc and scale on the whole real line. Write
u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
V = max(i/n-u_i) + max(u_i-(i-1)/n).
This is the unscaled sum of the two one-sided EDF distances.
Large values reject. The probability integral transform removes all
distribution parameters from the null law, not from the hypothesis.
The reference describes a general statistic, applied here through the
specified Student CDF, not a separately derived Student-specific test.

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
from pysatl_criterion.statistics.goodness_of_fit.student import KuiperStudentGofStatistic

statistic = KuiperStudentGofStatistic(df=5)
value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
print(value)
```

## Scientific source

M. A. Stephens (1974), EDF Statistics for Goodness of Fit and Some
Comparisons, JASA 69, 730-737, https://doi.org/10.1080/01621459.1974.10480196.

## Author(s)

Dmitriy Rusanov, Alexey Mironov
