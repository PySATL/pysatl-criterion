# Pearson chi-squared for equiprobable Student t bins.

## Hypothesis and formula

The null fixes df, loc and scale on the whole real line. Write
u_i = t_df.cdf((x_(i)-loc)/scale) for sorted observations, i=1,...,n.
For k=n_bins, use edges t_df.ppf(j/k), j=0,...,k on standardized
data, with infinite outer edges. Internal boundary ties enter the right
bin. Return sum((O_j-n/k)**2/(n/k)), where O_j are observed counts.
Empty cells contribute n/k. Under the null, counts are multinomial with
probabilities 1/k. The chi-squared(k-1) law is only asymptotic and needs
sufficiently large expected counts. Monte Carlo must retain n_bins.
Large values reject. The probability integral transform removes all
distribution parameters from the null law, not from the hypothesis.
The reference describes a general statistic, applied here through the
specified Student CDF, not a separately derived Student-specific test.

## Parameters

```text
parameters : ParameterValues
    Values with df, loc, scale fixed; omitted parameters are unknown.
n_bins : int, default: 10
    Fixed number of equiprobable bins, at least 2; booleans are rejected.
```

## Input and result

`execute_statistic(rvs, **kwargs)` accepts a finite, real, one-dimensional sample,
with at least one observation; ties and constants are allowed. It returns one scalar, leaves the input unchanged,
and ignores extra keyword arguments. Invalid samples and unrepresentable numerical
transformations raise `ValueError`. No preliminary fit or bootstrap is required.

## Example

```python
from pysatl_criterion.distribution.distributions import StudentDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.student import ChiSquareStudentGofStatistic

statistic = ChiSquareStudentGofStatistic(StudentDistributionDescriptor.DEFAULT.parse({'df': 5, 'loc': 0, 'scale': 1}))
value = statistic.execute_statistic([-2., -0.7, 0., 0.4, 1.8])
print(value)
```

## Scientific source

K. Pearson (1900), On the criterion that a given system of deviations
from the probable ... , https://doi.org/10.1080/14786440009463897.

## Author(s)

Dmitriy Rusanov, Alexey Mironov
