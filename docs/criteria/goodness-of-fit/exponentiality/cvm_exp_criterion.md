# Cramer-von Mises test for exponentiality

## Description
Performs Cramer-von Mises test for the hypothesis of exponentiality, see e.g. Cramer (1928), von Mises (1931), and Henze and Meintanis (2005).
The Cramer-von Mises test is a statistical goodness-of-fit test used to assess whether a given sample of data is consistent with an exponential distribution. This test is based on the empirical distribution function and measures the squared distance between the empirical cumulative distribution function and the theoretical cumulative distribution function of the exponential distribution.

Hypothesis of Exponentiality
The hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with rate parameter
lambda. In the implementation the rate parameter is supplied by the user and is used to evaluate the theoretical exponential cumulative distribution function.

Test Statistic
The Cramer-von Mises test statistic is based on the integrated squared difference between the empirical distribution function and the theoretical cumulative distribution function. Compared with the Kolmogorov-Smirnov statistic, which uses the largest absolute discrepancy, the Cramer-von Mises statistic accumulates discrepancies over the whole sample.

Calculate the Test Statistic: The observations are sorted, the exponential cumulative distribution function is evaluated at the ordered observations, and the squared deviations from the expected uniform plotting positions are summed.

Limitations
The classical Cramer-von Mises test is most direct when the distribution parameters are fixed in advance.

When the rate parameter is estimated from the same sample, the usual critical values are no longer directly applicable and adjusted critical values or simulation-based calibration may be required.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesExponentialityGofStatistic,
)


test_statistic = CramerVonMisesExponentialityGofStatistic(lam=0.5)
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`lam` - rate parameter of the exponential distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Cramer-von Mises test is an empirical distribution function test for exponentiality. The test statistic is

$$ W_n^2 = \frac{1}{12n} + \sum_{i=1}^{n} \left(F_0(X_{(i)}) - \frac{2i - 1}{2n}\right)^2 $$

where $X_{(i)}$ is the $i$-th order statistic, $n$ is the sample size, and $F_0$ is the cumulative distribution function of the reference exponential distribution.

For an exponential distribution with rate parameter $\lambda$, the cumulative distribution function is

$$ F_0(x) = 1 - exp(-\lambda x), \quad x \ge 0. $$

Large values of $W_n^2$ indicate larger deviations from the exponential model.

## Author(s)
Lev Golofastov

## References
Cramer, H. (1928): On the composition of elementary errors. - Skandinavisk Aktuarietidskrift, vol. 11, pp. 141-180.

von Mises, R. (1931): Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik. - Leipzig: Deuticke.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CramerVonMisesExponentialityGofStatistic,
)


test_statistic = CramerVonMisesExponentialityGofStatistic(lam=0.5)
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
