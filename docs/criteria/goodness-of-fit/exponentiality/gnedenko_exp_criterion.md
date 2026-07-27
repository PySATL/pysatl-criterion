# Gnedenko F-test of exponentiality

## Description
Performs Gnedenko F-test for the composite hypothesis of exponentiality, see e.g. Gnedenko, Belyayev and Solovyev (1969).
The Gnedenko F-test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by comparing groups of normalized spacings between ordered observations.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Gnedenko statistic is based on normalized spacings, so the unknown exponential scale is removed from the comparison.

Test Statistic
The Gnedenko F-test statistic is based on the fact that normalized spacings of exponential order statistics have a convenient distributional structure. The statistic compares the average of the first group of normalized spacings with the average of the remaining normalized spacings.

Calculate the Test Statistic: The observations are sorted after adding zero as the lower endpoint, normalized spacings are calculated, the spacings are split at index `r`, and the statistic is formed as a ratio of two spacing averages.

Limitations
The test assumes nonnegative lifetime-type observations.

The split point `r` can affect the sensitivity of the test. In the implementation the default value is `round(n / 2)`, where `n` is the sample size.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GnedenkoExponentialityGofStatistic,
)


test_statistic = GnedenkoExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

`r` - split point for the spacing ratio. Default value is `round(n / 2)`.

## Details

The Gnedenko F-test is a test for the composite hypothesis of exponentiality. Let
$X_{(1)} \leq \cdots \leq X_{(n)}$ denote the ordered sample and set $X_{(0)}=0$. Define the normalized spacings

$$ D_i = (n-i+1)(X_{(i)}-X_{(i-1)}), \quad i=1,\ldots,n. $$

The test statistic implemented here is

$$ GD_n = \frac{r^{-1}\sum_{i=1}^{r}D_i}{(n-r)^{-1}\sum_{i=r+1}^{n}D_i}. $$

Here $n$ is the sample size and $r$ is the split point. If `r` is not supplied, the implementation uses $r=round(n/2)$.

Large values of $GD_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Gnedenko, B.V., Belyayev, Yu.K. and Solovyev, A.D. (1969): Mathematical Methods of Reliability Theory. - New York and London: Academic Press.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GnedenkoExponentialityGofStatistic,
)


test_statistic = GnedenkoExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
