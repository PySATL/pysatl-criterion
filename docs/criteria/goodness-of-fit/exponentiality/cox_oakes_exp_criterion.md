# Test for exponentiality of Cox and Oakes

## Description
Performs Cox and Oakes test for the composite hypothesis of exponentiality, see e.g. Henze and Meintanis (2005, Sec. 2.5).
The Cox and Oakes test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution, which is a common model for lifetime, waiting-time, reliability, and survival data.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Cox and Oakes statistic uses normalized observations, so the unknown scale is removed by dividing each observation by the sample mean.

Test Statistic
The Cox and Oakes test statistic is based on a score-type approach for testing exponentiality. The statistic uses the transformed observations $Y_j=X_j/\overline{X}$ and combines them through the logarithmic term $(1-Y_j)\log(Y_j)$.

Calculate the Test Statistic: The observations are divided by the sample mean, the logarithmic score terms are summed, and the sample size is added to obtain the Cox and Oakes statistic.

Limitations
The test assumes positive lifetime-type observations, since the statistic uses logarithms of normalized data.

The asymptotic normal approximation may require moderate or large sample sizes; simulation-based calibration can be useful for small samples.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CoxOakesExponentialityGofStatistic,
)


test_statistic = CoxOakesExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Cox and Oakes test is a test for the composite hypothesis of exponentiality. The test statistic is

$$ CO_n = n + \sum_{j=1}^{n}(1-Y_j)\log(Y_j) $$

where

$$ Y_j = \frac{X_j}{\overline{X}}. $$

Here $n$ is the sample size and $\overline{X}$ is the sample mean. The statistic is scale invariant because it is calculated from the normalized observations $Y_j$.

Under the null hypothesis of exponentiality,

$$ \left(\frac{6}{n}\right)^{1/2}\frac{CO_n}{\pi} $$

is asymptotically standard normal (see, e.g., Henze and Meintanis (2005, Sec. 2.5)).

## Author(s)
Lev Golofastov

## References
Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

Cox, D.R. and Oakes, D. (1984): Analysis of Survival Data. - London: Chapman and Hall.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CoxOakesExponentialityGofStatistic,
)


test_statistic = CoxOakesExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
