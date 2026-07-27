# Hollander-Proshan test for exponentiality

## Description
Performs Hollander-Proshan test for the composite hypothesis of exponentiality, see e.g. Hollander and Proschan (1972).
The Hollander-Proshan test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by using a U-statistic based on the comparison of one lifetime with the sum of two other lifetimes.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Hollander-Proshan statistic is scale invariant because it uses only comparisons of sample observations and their sums.

Test Statistic
The Hollander-Proshan test statistic is based on the new-better-than-used property from reliability theory. For an exponential distribution, a new item is stochastically equivalent to a used item of any age. The statistic counts how often one observation is greater than the sum of two other observations.

Calculate the Test Statistic: For all triples of distinct observations with the second and third indices ordered as `j < k`, count the indicator of the event $X_i > X_j + X_k$ and normalize this count by the number of possible triples.

Limitations
The test assumes nonnegative lifetime-type observations.

The computation uses a triple loop over the sample and can be expensive for large samples.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    HollanderProshanExponentialityGofStatistic,
)


test_statistic = HollanderProshanExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Hollander-Proshan test is a test for the composite hypothesis of exponentiality. The test statistic implemented here is

$$ HP_n = \frac{2}{n(n-1)(n-2)} \sum_{i\ne j,\, i\ne k,\, j<k} I(X_i > X_j + X_k). $$

Here $n$ is the sample size and $I(\cdot)$ is the indicator function.

Under exponentiality,

$$ \sqrt{n}\left(HP_n-\frac{1}{4}\right) \xrightarrow{d} N\left(0,\frac{5}{432}\right). $$

Large values of $HP_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Hollander, M. and Proschan, F. (1972): Testing whether new is better than used. - Annals of Mathematical Statistics, vol. 43, no. 4, pp. 1136-1146. https://doi.org/10.1214/aoms/1177692466

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    HollanderProshanExponentialityGofStatistic,
)


test_statistic = HollanderProshanExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
