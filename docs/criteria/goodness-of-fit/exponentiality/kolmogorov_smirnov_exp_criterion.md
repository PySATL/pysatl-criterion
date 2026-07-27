# Test for exponentiality of Kolmogorov and Smirnov

## Description
Performs Kolmogorov-Smirnov test for the hypothesis of exponentiality.
The Kolmogorov-Smirnov test is a classical empirical distribution function test used to assess whether a given sample of data is consistent with a specified theoretical distribution. In the case of exponentiality, the test compares the empirical distribution function of the sample with the cumulative distribution function of the exponential distribution.

Hypothesis of Exponentiality
The hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution. The exponential distribution is widely used in reliability theory, survival analysis, queuing theory, and other fields where waiting times or lifetimes are modeled.

Test Statistic
The Kolmogorov-Smirnov test statistic is based on the maximum absolute difference between the empirical distribution function and the theoretical cumulative distribution function of the exponential distribution. Large values of the statistic indicate that the sample distribution differs from the exponential model.

Calculate the Test Statistic: The observations are sorted, the theoretical exponential cumulative distribution function is evaluated at the ordered observations, and the largest positive or negative deviation from the empirical distribution function is used as the test statistic.

Limitations
The classical Kolmogorov-Smirnov test is most direct when the distribution parameters are fixed in advance.

When parameters are estimated from the same sample, the usual Kolmogorov-Smirnov critical values are no longer directly applicable and adjusted critical values or simulation-based calibration may be required.

## Usage

## Arguments

## Details

The Kolmogorov-Smirnov test is an empirical distribution function test for exponentiality. The two-sided test statistic is

$$ D_n = \max(D_n^+, D_n^-) $$

where

$$ D_n^+ = \max_{1 \le i \le n} \left( \frac{i}{n} - F_0(X_{(i)}) \right) $$

and

$$ D_n^- = \max_{1 \le i \le n} \left( F_0(X_{(i)}) - \frac{i - 1}{n} \right). $$

Here $X_{(i)}$ is the $i$-th order statistic and $F_0$ is the cumulative distribution function of the reference exponential distribution. In the implementation, $F_0$ is the standard exponential cumulative distribution function.

For one-sided alternatives, the statistic is $D_n^+$ or $D_n^-$ respectively.

## Author(s)
Lev Golofastov

## References
Kolmogorov, A.N. (1933): Sulla determinazione empirica di una legge di distribuzione. — Giornale dell'Istituto Italiano degli Attuari, vol. 4, pp. 83-91.

Smirnov, N.V. (1948): Table for estimating the goodness of fit of empirical distributions. — Annals of Mathematical Statistics, vol. 19, pp. 279-281.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. — Metrika, vol. 61, pp. 29–45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovExponentialityGofStatistic,
)


test_statistic = KolmogorovSmirnovExponentialityGofStatistic(lam=0.5)
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
