# Atkinson test for exponentiality

## Description
Performs Atkinson test for the composite hypothesis of exponentiality, see e.g. Mimoto and Zitikis (2008).
The Atkinson test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is based on a moment ratio related to the Atkinson index and is designed to determine whether a given sample of data is consistent with an exponential distribution.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Atkinson statistic is scale invariant, so the unknown exponential scale is removed by dividing by the sample mean.

Test Statistic
The Atkinson test statistic compares a power mean of the sample with its arithmetic mean. Under exponentiality, the corresponding theoretical ratio is determined by the gamma function. The statistic measures the absolute difference between the empirical ratio and the theoretical exponential ratio.

Calculate the Test Statistic: The test statistic is calculated from the sample mean, the sample mean of the powered observations, and the value of the gamma function at `1 + p`.

Limitations
The choice of the power parameter `p` can affect the power of the test against different alternatives.

The asymptotic distribution is typically used for calibration, and simulation-based critical values may be preferable for small samples.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AtkinsonExponentialityGofStatistic,
)


test_statistic = AtkinsonExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

`p` - power parameter of the test statistic. Default value is `0.99`.

## Details

The Atkinson test is a test for the composite hypothesis of exponentiality. The test statistic is

$$ T_n(p) = \sqrt{n} \left| \frac{\left(n^{-1}\sum_{i=1}^{n}X_i^p\right)^{1/p}}{\overline{X}} - \left(\Gamma(1+p)\right)^{1/p} \right| $$

where $n$ is the sample size, $\overline{X}$ is the sample mean, $p$ is the power parameter, and $\Gamma(\cdot)$ is the gamma function.

Under exponentiality, the moment ratio

$$ \frac{(E X^p)^{1/p}}{E X} $$

is equal to $(\Gamma(1+p))^{1/p}$. Large values of $T_n(p)$ indicate larger deviations from the exponential model.

## Author(s)
Lev Golofastov

## References
Mimoto, N. and Zitikis, R. (2008): The Atkinson index, the Moran statistic, and testing exponentiality. - Journal of the Japan Statistical Society, vol. 38, no. 2, pp. 187-205.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AtkinsonExponentialityGofStatistic,
)


test_statistic = AtkinsonExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
