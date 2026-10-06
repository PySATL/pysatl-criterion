# Cramer-von Mises test for beta distribution

## Description
Performs Cramer-von Mises goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic accumulates squared differences between the empirical distribution function and the theoretical beta cumulative distribution function.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

Test Statistic
The observations are sorted, transformed by the reference beta cumulative distribution function, and compared with expected uniform plotting positions.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CrammerVonMisesBetaGofStatistic,
)


test_statistic = CrammerVonMisesBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
The Cramer-von Mises statistic is

$$ W_n^2 = \frac{1}{12n} + \sum_{i=1}^{n} \left(F_0(X_{(i)}) - \frac{2i - 1}{2n}\right)^2 $$

where $F_0$ is the reference beta cumulative distribution function.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References

The reference studies the one-sample continuous-null W-squared statistic. This class applies it to a specified Beta CDF; neither shape is estimated.

S. Csorgo and J. J. Faraway (1996), "The Exact and Asymptotic
   Distributions of Cramer-von Mises Statistics", JRSS B 58, 221-234.
   https://doi.org/10.1111/j.2517-6161.1996.tb02077.x

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CrammerVonMisesBetaGofStatistic,
)


test_statistic = CrammerVonMisesBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
