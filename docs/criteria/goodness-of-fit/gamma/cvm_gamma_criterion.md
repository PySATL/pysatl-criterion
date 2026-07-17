# Cramer-von Mises test for gamma distribution

## Description
Performs Cramer-von Mises goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic accumulates squared differences between empirical plotting positions and gamma cumulative distribution function values.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesGammaGofStatistic,
)


test_statistic = CramerVonMisesGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The Cramer-von Mises statistic is

$$ W_n^2 = \frac{1}{12n} + \sum_{i=1}^{n} \left(F_0(X_{(i)}) - \frac{2i - 1}{2n}\right)^2 $$

where $F_0$ is the reference gamma cumulative distribution function.

## Author(s)
Alexey Mironov

## References
Cramer, H. (1928): On the composition of elementary errors. - Skandinavisk Aktuarietidskrift, vol. 11, pp. 141-180.

von Mises, R. (1931): Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik. - Leipzig: Deuticke.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesGammaGofStatistic,
)


test_statistic = CramerVonMisesGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
