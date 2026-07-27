# WE test for exponentiality

## Description
Performs WE test for the composite hypothesis of exponentiality, see e.g. Ascher (1990).
The WE test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is based on the squared coefficient of variation, a scale-free measure of dispersion that has a characteristic value for an exponential distribution.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The WE statistic is scale invariant because it is calculated from the ratio of the sample variance to the squared sample mean.

Test Statistic
The WE test statistic is based on the coefficient of variation. For an exponential distribution, the population coefficient of variation is equal to one, so departures from this value indicate departures from exponentiality.

Calculate the Test Statistic: The sample mean and population-form sample variance are calculated, the variance is divided by the squared mean, and the result is multiplied by the factor `(n - 1) / n**2`.

Limitations
The test assumes nonnegative lifetime-type observations.

The coefficient of variation can be strongly affected by outlying observations and by samples with a small mean.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WeExponentialityGofStatistic,
)


test_statistic = WeExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The WE test is a test for the composite hypothesis of exponentiality. The test statistic implemented here is

$$ WE_n = \frac{(n-1)\widehat{\sigma}_n^2}{n^2\overline{X}^2}, $$

where $n$ is the sample size, $\overline{X}$ is the sample mean, and

$$ \widehat{\sigma}_n^2 = \frac{1}{n}\sum_{i=1}^{n}(X_i-\overline{X})^2. $$

Equivalently,

$$ WE_n = \frac{n-1}{n^2}CV_n^2, $$

where $CV_n^2$ is the squared sample coefficient of variation calculated with the population-form variance.

Large values of $WE_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Ascher, S. (1990): A survey of tests for exponentiality. - Communications in Statistics - Theory and Methods, vol. 19, pp. 1811-1825.

Holland, B. (1989): Monte Carlo simulation of the power of a test for the exponential distribution of survival times. - Computer Methods and Programs in Biomedicine, vol. 29, no. 4, pp. 245-250. https://doi.org/10.1016/0169-2607(89)90109-0

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WeExponentialityGofStatistic,
)


test_statistic = WeExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
