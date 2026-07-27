# Lorenz test for exponentiality

## Description
Performs Lorenz test for the composite hypothesis of exponentiality, see e.g. Gail and Gastwirth (1978).
The Lorenz test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by comparing the sample Lorenz curve with the Lorenz curve of the exponential distribution.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Lorenz statistic is scale invariant because it is calculated as a ratio of partial sums to the sample total.

Test Statistic
The Lorenz test statistic is based on the sample Lorenz curve. For a fixed value `p`, the statistic measures the share of the sample total contributed by the smallest fraction `p` of the ordered observations.

Calculate the Test Statistic: The observations are sorted, the first `int(n * p)` ordered observations are summed, and this partial sum is divided by the full sample sum.

Limitations
The test assumes nonnegative lifetime-type observations.

The choice of the Lorenz curve point `p` can affect the power of the test. In the implementation the default value is `0.5`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LorenzExponentialityGofStatistic,
)


test_statistic = LorenzExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

`p` - Lorenz curve point used by the statistic. Default value is `0.5`.

## Details

The Lorenz test is a test for the composite hypothesis of exponentiality. Let
$X_{(1)} \leq \cdots \leq X_{(n)}$ denote the ordered sample. The test statistic implemented here is

$$ L_n(p) = \frac{\sum_{i=1}^{\lfloor np \rfloor}X_{(i)}}{\sum_{i=1}^{n}X_i}. $$

Here $n$ is the sample size and $p$ is the Lorenz curve point.

For an exponential distribution, the population Lorenz curve is

$$ L(p) = p + (1-p)\log(1-p), \quad 0 < p < 1. $$

The statistic is scale free because both the numerator and denominator are multiplied by the same scale factor when the data are rescaled.

Large values of $L_n(p)$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Gail, M.H. and Gastwirth, J.L. (1978): A scale-free goodness-of-fit test for the exponential distribution based on the Lorenz curve. - Journal of the American Statistical Association, vol. 73, no. 364, pp. 787-793. https://doi.org/10.1080/01621459.1978.10480100

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LorenzExponentialityGofStatistic,
)


test_statistic = LorenzExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
