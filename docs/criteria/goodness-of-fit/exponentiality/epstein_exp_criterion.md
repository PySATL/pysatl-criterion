# Epstein test for exponentiality

## Description
Performs Epstein test for the composite hypothesis of exponentiality, see Epstein (1960a, 1960b).
The Epstein test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is based on normalized spacings between ordered observations and is designed to determine whether a given sample of lifetime data is consistent with an exponential distribution.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Epstein statistic uses spacings between order statistics in a way that removes the unknown scale of the exponential distribution.

Test Statistic
The Epstein test statistic is based on the normalized spacings of the ordered sample. For exponential data, these normalized spacings have a structure that can be used to construct a goodness-of-fit statistic. Large values of the statistic indicate departures from exponentiality.

Calculate the Test Statistic: The observations are sorted, the spacings between consecutive order statistics are multiplied by decreasing weights, and the statistic compares the logarithm of the arithmetic mean of these weighted spacings with the mean of their logarithms.

Limitations
The test assumes positive lifetime-type observations, since the statistic uses logarithms of normalized spacings.

The chi-square approximation may require moderate or large sample sizes; small-sample calibration can require simulation or tabulated critical values.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    EpsteinExponentialityGofStatistic,
)


test_statistic = EpsteinExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Epstein test is a test for the composite hypothesis of exponentiality. The test statistic is

$$ EPS_n = \frac{2n\left(\log\left(n^{-1}\sum_{i=1}^{n}D_i\right) - n^{-1}\sum_{i=1}^{n}\log(D_i)\right)}{1+(n+1)/(6n)} $$

where

$$ D_i = (n-i+1)(X_{(i)} - X_{(i-1)}), \quad X_{(0)} = 0, $$

and $X_{(1)} \le \cdots \le X_{(n)}$ are the order statistics.

Under exponentiality, $EPS_n$ is approximately distributed as a chi-square random variable with $n-1$ degrees of freedom.

## Author(s)
Lev Golofastov

## References
Epstein, B. (1960a): Tests for the validity of the assumption that the underlying distribution of life is exponential. Part I. - Technometrics, vol. 2, no. 1, pp. 83-101.

Epstein, B. (1960b): Tests for the validity of the assumption that the underlying distribution of life is exponential. Part II. - Technometrics, vol. 2, no. 2, pp. 167-183.

Ascher, S. (1990): A survey of tests for exponentiality. - Communications in Statistics - Theory and Methods, vol. 19, pp. 1811-1825.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    EpsteinExponentialityGofStatistic,
)


test_statistic = EpsteinExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
