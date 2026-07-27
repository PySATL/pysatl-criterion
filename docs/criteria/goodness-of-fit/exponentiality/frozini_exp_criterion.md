# Frozini test for exponentiality

## Description
Performs Frozini test for the composite hypothesis of exponentiality, see Frozini (1987).
The Frozini test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by comparing the empirical distribution function with the fitted exponential distribution function.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Frozini statistic estimates the unknown scale by the sample mean and then compares the fitted exponential cumulative distribution function with empirical plotting positions.

Test Statistic
The Frozini test statistic is based on absolute deviations between the fitted exponential cumulative distribution function and the expected empirical distribution function positions. Large values of the statistic indicate larger departures from exponentiality.

Calculate the Test Statistic: The observations are sorted, the exponential cumulative distribution function with mean equal to the sample mean is evaluated at each order statistic, and the absolute differences from $(i-0.5)/n$ are summed and normalized by $\sqrt{n}$.

Limitations
The test is calibrated by the null distribution of the statistic, which is commonly obtained by Monte Carlo simulation.

The statistic assumes lifetime-type observations and uses the fitted exponential distribution with scale equal to the sample mean.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    FroziniExponentialityGofStatistic,
)


test_statistic = FroziniExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Frozini test is a test for the composite hypothesis of exponentiality. The test statistic is

$$ B_n = \frac{1}{\sqrt{n}}\sum_{i=1}^{n}\left|1 - exp\left(-\frac{X_{(i)}}{\overline{X}}\right) - \frac{i - 0.5}{n}\right| $$

where $X_{(i)}$ is the $i$-th order statistic, $n$ is the sample size, and $\overline{X}$ is the sample mean.

The term

$$ 1 - exp\left(-\frac{X_{(i)}}{\overline{X}}\right) $$

is the cumulative distribution function of the fitted exponential distribution evaluated at the ordered observation $X_{(i)}$.

## Author(s)
Lev Golofastov

## References
Frozini, B.V. (1987): On the distribution and power of a goodness-of-fit statistic with parametric and nonparametric applications. In: Revesz, P., Sarkadi, K. and Sen, P.K. (eds.) Goodness-of-fit. - Amsterdam-Oxford-New York: North-Holland, pp. 133-154.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    FroziniExponentialityGofStatistic,
)


test_statistic = FroziniExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
