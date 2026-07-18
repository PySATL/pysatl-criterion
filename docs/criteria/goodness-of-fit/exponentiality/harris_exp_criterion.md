# Harris modification of Gnedenko F-test

## Description
Performs Harris modification of Gnedenko F-test for the composite hypothesis of exponentiality, see e.g. Harris (1976).
The Harris modification is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by comparing tail groups of normalized spacings with the middle group of normalized spacings.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Harris statistic is based on normalized spacings, so the unknown exponential scale is removed from the comparison.

Test Statistic
The Harris modification improves the Gnedenko F-test by using a symmetric spacing comparison. Instead of comparing only the first group of spacings with the remaining spacings, it combines the first and last `r` normalized spacings and compares their average with the average of the middle spacings.

Calculate the Test Statistic: The observations are sorted after adding zero as the lower endpoint, normalized spacings are calculated, the first and last `r` spacings are averaged, and the result is divided by the average of the middle spacings.

Limitations
The test assumes nonnegative lifetime-type observations.

The split point `r` can affect the sensitivity of the test. In the implementation the default value is `round(n / 4)`, where `n` is the sample size.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    HarrisExponentialityGofStatistic,
)


test_statistic = HarrisExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

`r` - split point for the symmetric spacing comparison. Default value is `round(n / 4)`.

## Details

The Harris modification of Gnedenko F-test is a test for the composite hypothesis of exponentiality. Let
$X_{(1)} \leq \cdots \leq X_{(n)}$ denote the ordered sample and set $X_{(0)}=0$. Define the normalized spacings

$$ D_i = (n-i+1)(X_{(i)}-X_{(i-1)}), \quad i=1,\ldots,n. $$

The test statistic implemented here is

$$ HM_n = \frac{(2r)^{-1}\left(\sum_{i=1}^{r}D_i+\sum_{i=n-r+1}^{n}D_i\right)}{(n-2r)^{-1}\sum_{i=r+1}^{n-r}D_i}. $$

Here $n$ is the sample size and $r$ is the split point. If `r` is not supplied, the implementation uses $r=round(n/4)$.

Under exponentiality, the statistic has an F distribution with $4r$ and $2(n-2r)$ degrees of freedom (see, e.g., Harris (1976)).

Large values of $HM_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Harris, C.M. (1976): A note on testing for exponentiality. - Naval Research Logistics Quarterly, vol. 23, pp. 169-175.

Ascher, S. (1990): A survey of tests for exponentiality. - Communications in Statistics - Theory and Methods, vol. 19, pp. 1811-1825.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    HarrisExponentialityGofStatistic,
)


test_statistic = HarrisExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
