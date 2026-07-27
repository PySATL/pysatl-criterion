# Test for exponentiality of Wong and Wong

## Description
Performs Wong and Wong test for the composite hypothesis of exponentiality, see e.g. Wong and Wong (1979).
The Wong and Wong test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by using the extremal quotient of the sample.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Wong and Wong statistic is scale invariant because it is calculated as a ratio of the largest and smallest order statistics.

Test Statistic
The Wong and Wong test statistic is based on the extremal quotient. It compares the maximum observation with the minimum observation and uses the resulting ratio as evidence about departure from exponentiality.

Calculate the Test Statistic: The smallest and largest observations are found, and the largest observation is divided by the smallest observation.

Limitations
The test assumes strictly positive lifetime-type observations because the sample minimum is used as the denominator.

The statistic depends only on the two extreme observations and can be sensitive to outliers or measurement errors at the sample extremes.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WongWongExponentialityGofStatistic,
)


test_statistic = WongWongExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Wong and Wong test is a test for the composite hypothesis of exponentiality. Let
$X_{(1)} \leq \cdots \leq X_{(n)}$ denote the ordered sample. The test statistic implemented here is

$$ WW_n = \frac{X_{(n)}}{X_{(1)}}. $$

Here $n$ is the sample size, $X_{(1)}$ is the sample minimum, and $X_{(n)}$ is the sample maximum.

The statistic is scale free because multiplying all observations by the same positive constant leaves $WW_n$ unchanged.

Large values of $WW_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Wong, P.G. and Wong, S.P. (1979): An extremal quotient test for exponential distributions. - Metrika, vol. 26, pp. 1-4.

Ascher, S. (1990): A survey of tests for exponentiality. - Communications in Statistics - Theory and Methods, vol. 19, pp. 1811-1825.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WongWongExponentialityGofStatistic,
)


test_statistic = WongWongExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
