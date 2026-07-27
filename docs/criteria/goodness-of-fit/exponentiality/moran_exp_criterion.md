# Moran test for exponentiality

## Description
Performs Moran test for the composite hypothesis of exponentiality, see e.g. Moran (1951).
The Moran test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by using a logarithmic ratio of the geometric mean to the arithmetic mean.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Moran statistic is scale invariant because it is calculated from the normalized observations $X_i/\overline{X}$.

Test Statistic
The Moran test statistic is based on a logarithmic moment identity for the exponential distribution. If the data are exponential, then the expected value of $\log(X/\mu)$ is $-\gamma$, where $\gamma$ is the Euler-Mascheroni constant.

Calculate the Test Statistic: The observations are divided by the sample mean, their logarithms are averaged, and the Euler-Mascheroni constant is added.

Limitations
The test assumes strictly positive lifetime-type observations because logarithms are used.

The statistic can be sensitive to rounding, truncation, or very small observations.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MoranExponentialityGofStatistic,
)


test_statistic = MoranExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Moran test is a test for the composite hypothesis of exponentiality. The test statistic implemented here is

$$ MN_n = \gamma + \frac{1}{n}\sum_{i=1}^{n}\log\left(\frac{X_i}{\overline{X}}\right), $$

where $n$ is the sample size, $\overline{X}$ is the sample mean, and $\gamma$ is the Euler-Mascheroni constant.

Equivalently, using the geometric mean $G_n=(\prod_{i=1}^{n}X_i)^{1/n}$,

$$ MN_n = \gamma + \log\left(\frac{G_n}{\overline{X}}\right). $$

Under exponentiality,

$$ \sqrt{n}MN_n \xrightarrow{d} N\left(0,\frac{\pi^2}{6}-1\right). $$

Large values of $MN_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Moran, P.A.P. (1951): The random division of an interval - Part II. - Journal of the Royal Statistical Society. Series B (Methodological), vol. 13, pp. 147-150.

Tchirina, A.V. (2005): Bahadur efficiency and local optimality of a test for exponentiality based on the Moran statistics. - Journal of Mathematical Sciences, vol. 127, no. 1, pp. 1812-1819.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MoranExponentialityGofStatistic,
)


test_statistic = MoranExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
