# Test for exponentiality based on the Pietra statistic

## Description
Performs a test for the composite hypothesis of exponentiality based on the Pietra statistic, see e.g. Ascher (1990).
The Pietra test is a statistical goodness-of-fit test used to assess whether a given sample of data is consistent with an exponential distribution. This test is based on the Pietra index, also known as half the relative mean absolute deviation.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Pietra statistic is scale invariant because it divides the mean absolute deviation from the sample mean by the sample mean.

Test Statistic
The Pietra test statistic is based on the average absolute deviation of observations from the sample mean. Under exponentiality, the corresponding population quantity has a fixed value that does not depend on the unknown scale parameter.

Calculate the Test Statistic: The sample mean is calculated, absolute deviations from that mean are summed, and the result is normalized by twice the sample size and the sample mean.

Limitations
The test assumes nonnegative lifetime-type observations.

The statistic is a single dispersion measure, so it may be less sensitive to departures that do not strongly change relative mean absolute deviation.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    PietraExponentialityGofStatistic,
)


test_statistic = PietraExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Pietra test is a test for the composite hypothesis of exponentiality. The test statistic implemented here is

$$ PT_n = \frac{\sum_{i=1}^{n}|X_i-\overline{X}|}{2n\overline{X}}, $$

where $n$ is the sample size and $\overline{X}$ is the sample mean.

Equivalently, $PT_n$ is the sample mean absolute deviation about the mean divided by twice the sample mean.

For an exponential distribution,

$$ PT = \frac{1}{e}. $$

The implemented test uses a two-sided alternative, so values of $PT_n$ that are too small or too large relative to the exponential reference distribution are evidence against exponentiality.

## Author(s)
Lev Golofastov

## References
Pietra, G. (1915): Delle relazioni tra gli indici di variabilita (Nota I). - Atti del Reale Istituto Veneto di Scienze, Lettere ed Arti, vol. 74, pp. 775-792.

Ascher, S. (1990): A survey of tests for exponentiality. - Communications in Statistics - Theory and Methods, vol. 19, pp. 1811-1825.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    PietraExponentialityGofStatistic,
)


test_statistic = PietraExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
