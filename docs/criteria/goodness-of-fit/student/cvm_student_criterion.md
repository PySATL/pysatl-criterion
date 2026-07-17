# Cramer-von Mises test for Student distribution

## Description
Performs Cramer-von Mises goodness-of-fit test for the hypothesis that the sample comes from a Student's t-distribution.
The statistic accumulates squared differences between empirical plotting positions and Student cumulative distribution function values.

Hypothesis of Student Distribution
The null hypothesis is that the data comes from a Student's t-distribution with positive degrees of freedom `df`, location parameter `loc`, and positive scale parameter `scale`.

Test Statistic
The observations are sorted, standardized, transformed by the Student cumulative distribution function, and compared with expected uniform plotting positions.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesStudentGofStatistic,
)


test_statistic = CramerVonMisesStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```

## Arguments
`df` - positive degrees of freedom of the Student's t-distribution. Default value is `1`.

`loc` - location parameter of the Student's t-distribution. Default value is `0`.

`scale` - positive scale parameter of the Student's t-distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The Cramer-von Mises statistic is

$$ W_n^2 = \frac{1}{12n} + \sum_{i=1}^{n} \left(F_0(X_{(i)}) - \frac{2i - 1}{2n}\right)^2 $$

where $F_0$ is the reference Student cumulative distribution function after standardization.

## Author(s)
Alexey Mironov

## References
Cramer, H. (1928): On the composition of elementary errors. - Skandinavisk Aktuarietidskrift, vol. 11, pp. 141-180.

von Mises, R. (1931): Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik. - Leipzig: Deuticke.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesStudentGofStatistic,
)


test_statistic = CramerVonMisesStudentGofStatistic(df=5, loc=0, scale=1)
statistic_result = test_statistic.execute_statistic([-1.8, -0.9, -0.25, 0.0, 0.31, 0.95, 1.7])
print(statistic_result)
```
