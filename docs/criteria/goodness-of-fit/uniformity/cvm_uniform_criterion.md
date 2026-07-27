# Cramer-von Mises test for uniformity

## Description
Performs the Cramer-von Mises goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic accumulates squared differences between the empirical distribution function and the theoretical uniform cumulative distribution function.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

Test Statistic
The observations are sorted, transformed by the reference uniform cumulative distribution function, and compared with expected uniform plotting positions.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CrammerVonMisesUniformGofStatistic,
)


test_statistic = CrammerVonMisesUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The Cramer-von Mises statistic is

$$ W_n^2 = \frac{1}{12n} + \sum_{i=1}^{n} \left(F_0(X_{(i)}) - \frac{2i - 1}{2n}\right)^2 $$

where $X_{(i)}$ is the $i$-th order statistic and $F_0$ is the reference uniform cumulative distribution function.
Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Cramer, H. (1928): On the composition of elementary errors. - Skandinavisk Aktuarietidskrift, vol. 11, pp. 141-180.

von Mises, R. (1931): Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik. - Leipzig: Deuticke.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CrammerVonMisesUniformGofStatistic,
)


test_statistic = CrammerVonMisesUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
