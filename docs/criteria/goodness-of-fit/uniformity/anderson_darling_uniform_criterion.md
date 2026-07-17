# Anderson-Darling test for uniformity

## Description
Performs the Anderson-Darling goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The test gives more weight to discrepancies in the tails than the Cramer-von Mises statistic.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

Test Statistic
The statistic is computed from the ordered sample using the log cumulative distribution function and log survival function of the reference uniform distribution.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingUniformGofStatistic,
)


test_statistic = AndersonDarlingUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For the uniform distribution on $[a, b]$,

$$ F_0(x) = \frac{x - a}{b - a}, \quad a \le x \le b. $$

The implementation evaluates `logcdf` and `logsf` for ordered observations and passes them to the common Anderson-Darling statistic implementation.
Large values indicate stronger deviation from the uniform model.

## Author(s)
Alexey Mironov

## References
Anderson, T.W. and Darling, D.A. (1952): Asymptotic theory of certain goodness of fit criteria based on stochastic processes. - Annals of Mathematical Statistics, vol. 23, pp. 193-212.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AndersonDarlingUniformGofStatistic,
)


test_statistic = AndersonDarlingUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
