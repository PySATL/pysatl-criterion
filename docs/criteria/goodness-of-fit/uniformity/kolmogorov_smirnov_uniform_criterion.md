# Kolmogorov-Smirnov test for uniformity

## Description
Performs the Kolmogorov-Smirnov goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic measures the largest discrepancy between the empirical distribution function and the cumulative distribution function of the reference uniform distribution.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.
All observations passed to `execute_statistic` must lie inside this interval.

Test Statistic
The statistic is based on the maximum vertical distance between the empirical cumulative distribution function and the theoretical uniform cumulative distribution function.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovUniformGofStatistic,
)


test_statistic = KolmogorovSmirnovUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`alternative_type` - alternative hypothesis type used by the Kolmogorov-Smirnov statistic.

`mode` - calculation mode passed to the Kolmogorov-Smirnov statistic. Default value is `auto`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For ordered observations $X_{(i)}$, the implementation evaluates

$$ F_0(X_{(i)}) = \frac{X_{(i)} - a}{b - a} $$

and passes these values to the common Kolmogorov-Smirnov statistic implementation.
Large values indicate stronger deviation from the uniform model.

## Author(s)
Alexey Mironov

## References
Kolmogorov, A.N. (1933): Sulla determinazione empirica di una legge di distribuzione. - Giornale dell'Istituto Italiano degli Attuari, vol. 4, pp. 83-91.

Smirnov, N.V. (1948): Table for estimating the goodness of fit of empirical distributions. - Annals of Mathematical Statistics, vol. 19, pp. 279-281.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovUniformGofStatistic,
)


test_statistic = KolmogorovSmirnovUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
