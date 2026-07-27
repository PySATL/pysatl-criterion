# Kochar test for exponentiality

## Description
Performs Kochar test for the composite hypothesis of exponentiality, see e.g. Kochar (1985).
The Kochar test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution against monotone failure rate average alternatives.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Kochar statistic is scale invariant because it is normalized by the sample total.

Test Statistic
The Kochar test statistic is based on a weighted linear combination of order statistics. The weights are defined by a score function evaluated at the plotting positions $i/(n+1)$, and the resulting weighted sum is standardized to have an asymptotic standard normal distribution under exponentiality.

Calculate the Test Statistic: The observations are sorted, Kochar scores are calculated at the plotting positions, the weighted sum of the ordered observations is divided by the sample total, and the result is multiplied by the normalizing factor.

Limitations
The test assumes nonnegative lifetime-type observations.

The test is designed for monotone failure rate average alternatives, so it may be less sensitive to other departures from exponentiality.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KocharExponentialityGofStatistic,
)


test_statistic = KocharExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Kochar test is a test for the composite hypothesis of exponentiality. Let
$X_{(1)} \leq \cdots \leq X_{(n)}$ denote the ordered sample. The test statistic implemented here is

$$ KC_n = \sqrt{\frac{108n}{17}}\frac{\sum_{i=1}^{n}J(i/(n+1))X_{(i)}}{\sum_{i=1}^{n}X_i}, $$

where

$$ J(u) = 2(1-u)(1-\log(1-u)) - 1. $$

Here $n$ is the sample size. Under the null hypothesis of exponentiality, $KC_n$ is asymptotically standard normal.

Large values of $KC_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Kochar, S.C. (1985): Testing exponenttality against monotone failure rate average. - Communications in Statistics - Theory and Methods, vol. 14, no. 2, pp. 381-392. https://doi.org/10.1080/03610928508828919

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KocharExponentialityGofStatistic,
)


test_statistic = KocharExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
