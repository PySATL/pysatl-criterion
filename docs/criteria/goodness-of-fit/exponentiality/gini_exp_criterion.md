# Test for exponentiality based on the Gini statistic

## Description
Performs the Gini test for the composite hypothesis of exponentiality, see e.g. Gail and Gastwirth (1978) and Henze and Meintanis (2005).
The Gini test is a statistical goodness-of-fit test used to assess whether a given sample of data is consistent with an exponential distribution. This test is based on the sample Gini index, a scale-free measure constructed from the ordered spacings of the observations.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Gini statistic is scale invariant because it is normalized by the sample total, so the unknown exponential scale does not have to be specified.

Test Statistic
The Gini test statistic is based on Gini's mean difference and the associated Gini index. For exponential data the population Gini index is equal to one half, so deviations of the sample Gini index from this value indicate departures from exponentiality.

Calculate the Test Statistic: The observations are sorted, consecutive spacings are weighted by the products of the numbers of observations below and above each spacing, and the result is normalized by the sample total.

Limitations
The test is intended for nonnegative lifetime-type observations.

The statistic is simple to compute, but critical values or p-values should be obtained from the appropriate exact, asymptotic, or simulation-based distribution for the chosen sample size.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GiniExponentialityGofStatistic,
)


test_statistic = GiniExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Gini test is a test for the composite hypothesis of exponentiality. Let
$X_{(1)} \leq \cdots \leq X_{(n)}$ denote the ordered sample. The statistic implemented here is

$$ G_n = \frac{\sum_{i=1}^{n-1} i(n-i)(X_{(i+1)}-X_{(i)})}{(n-1)\sum_{i=1}^{n}X_i}. $$

Here $n$ is the sample size. The numerator is a weighted sum of adjacent ordered spacings, and the denominator makes the statistic scale free.

For an exponential distribution,

$$ G = \frac{1}{2}. $$

The implemented test uses a two-sided alternative, so values of $G_n$ that are too small or too large relative to the exponential reference distribution are evidence against exponentiality.

## Author(s)
Lev Golofastov

## References
Gail, M.H. and Gastwirth, J.L. (1978): A scale-free goodness-of-fit test for the exponential distribution based on the Gini statistic. - Journal of the Royal Statistical Society: Series B (Methodological), vol. 40, no. 3, pp. 350-357. https://doi.org/10.1111/j.2517-6161.1978.tb01048.x

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GiniExponentialityGofStatistic,
)


test_statistic = GiniExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
