# Deshpande test for exponentiality

## Description
Performs Deshpande test for the composite hypothesis of exponentiality, see Deshpande (1983).
The Deshpande test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution against aging-type alternatives such as increasing failure rate average alternatives.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The statistic is based on pairwise comparisons and is scale invariant for positive rescaling of the data.

Test Statistic
The Deshpande test statistic is a U-statistic based on pairwise comparisons of observations. It counts how often one observation is larger than a fixed multiple of another observation. The multiplier is controlled by the parameter `b`.

Calculate the Test Statistic: For every ordered pair of distinct observations, the indicator of the event $X_i > bX_j$ is evaluated, and the average of these indicators is used as the statistic.

Limitations
The value of the parameter `b` can affect the power of the test against different alternatives.

The direct implementation uses all ordered pairs of observations, so the computational cost grows quadratically with the sample size.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DeshpandeExponentialityGofStatistic,
)


test_statistic = DeshpandeExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

`b` - threshold parameter for the pairwise comparison. Default value is `0.44`.

## Details

The Deshpande test is a test for the composite hypothesis of exponentiality. The test statistic is

$$ J_n = \frac{1}{n(n-1)} \sum_{i \ne j} I(X_i > bX_j) $$

where $I(\cdot)$ is the indicator function, $n$ is the sample size, and $b$ is a positive parameter of the test.

Under exponentiality,

$$ \sqrt{n}\left(J_n - \frac{1}{b+1}\right) $$

is asymptotically normal with variance depending on $b$ (see Deshpande (1983)).

## Author(s)
Lev Golofastov

## References
Deshpande, J.V. (1983): A class of tests for exponentiality against increasing failure rate average alternatives. - Biometrika, vol. 70, pp. 514-518.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    DeshpandeExponentialityGofStatistic,
)


test_statistic = DeshpandeExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
