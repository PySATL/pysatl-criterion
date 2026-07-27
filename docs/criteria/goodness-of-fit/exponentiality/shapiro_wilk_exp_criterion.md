# Shapiro-Wilk test for exponentiality

## Description
Performs Shapiro-Wilk test for the composite hypothesis of exponentiality, see e.g. Shapiro and Wilk (1972).
The Shapiro-Wilk test for exponentiality is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test adapts the analysis-of-variance idea behind the Shapiro-Wilk normality test to the exponential distribution.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Shapiro-Wilk exponentiality statistic is scale invariant because it compares the squared distance between the sample mean and the sample minimum with the sample variance around the mean.

Test Statistic
The Shapiro-Wilk exponentiality test statistic is a W-type statistic for complete samples from an exponential distribution. It is based on the ratio of a squared linear estimate of the scale to the usual sum of squares about the sample mean.

Calculate the Test Statistic: The observations are sorted, the sample mean is calculated, the squared difference between the sample mean and the smallest observation is multiplied by the sample size, and the result is divided by the centered sum of squares multiplied by `n - 1`.

Limitations
The test assumes nonnegative lifetime-type observations.

The statistic requires at least two distinct observations because the denominator is based on the centered sum of squares.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ShapiroWilkExponentialityGofStatistic,
)


test_statistic = ShapiroWilkExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Shapiro-Wilk test is a test for the composite hypothesis of exponentiality. Let $X_{(1)}$ denote the sample minimum. The test statistic implemented here is

$$ SW_n = \frac{n(\overline{X}-X_{(1)})^2}{(n-1)\sum_{i=1}^{n}(X_i-\overline{X})^2}, $$

where $n$ is the sample size and $\overline{X}$ is the sample mean.

The statistic is scale invariant and is bounded between its lower endpoint and 1 for nonconstant samples.

The implementation uses a left-sided alternative, so small values of $SW_n$ indicate larger deviations from the exponential model in the direction used by this statistic.

## Author(s)
Lev Golofastov

## References
Shapiro, S.S. and Wilk, M.B. (1972): An analysis of variance test for the exponential distribution (complete samples). - Technometrics, vol. 14, no. 2, pp. 355-370. https://doi.org/10.1080/00401706.1972.10488921

Shapiro, S.S. and Wilk, M.B. (1965): An analysis of variance test for normality (complete samples). - Biometrika, vol. 52, no. 3/4, pp. 591-611.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ShapiroWilkExponentialityGofStatistic,
)


test_statistic = ShapiroWilkExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
