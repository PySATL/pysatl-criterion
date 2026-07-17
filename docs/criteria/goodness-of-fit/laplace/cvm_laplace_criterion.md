# Cramer-von Mises test for Laplace distribution

## Description
Performs Cramer-von Mises goodness-of-fit test for the hypothesis that the sample comes from a Laplace distribution.
The statistic accumulates squared differences between empirical plotting positions and Laplace cumulative distribution function values.

Hypothesis of Laplace Distribution
The null hypothesis is that the data comes from a Laplace distribution with location parameter `t` and positive scale parameter `s`.

Test Statistic
The observations are sorted, transformed by the reference Laplace cumulative distribution function, and compared with expected uniform plotting positions.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesLaplaceGofStatistic,
)


test_statistic = CramerVonMisesLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```

## Arguments
`t` - location parameter of the Laplace distribution. Default value is `0.0`.

`s` - positive scale parameter of the Laplace distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The Cramer-von Mises statistic is

$$ W_n^2 = \frac{1}{12n} + \sum_{i=1}^{n} \left(F_0(X_{(i)}) - \frac{2i - 1}{2n}\right)^2 $$

where $F_0$ is the reference Laplace cumulative distribution function.

## Author(s)
Alexey Mironov

## References
Cramer, H. (1928): On the composition of elementary errors. - Skandinavisk Aktuarietidskrift, vol. 11, pp. 141-180.

von Mises, R. (1931): Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik. - Leipzig: Deuticke.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesLaplaceGofStatistic,
)


test_statistic = CramerVonMisesLaplaceGofStatistic(t=0, s=1)
statistic_result = test_statistic.execute_statistic([-1.7, -0.9, -0.35, 0.0, 0.42, 0.88, 1.64])
print(statistic_result)
```
