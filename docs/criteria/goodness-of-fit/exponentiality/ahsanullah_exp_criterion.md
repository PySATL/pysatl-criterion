# Test for exponentiality based on Ahsanullah characterization

## Description
Performs Ahsanullah test for the hypothesis of exponentiality, see e.g. Ahsanullah and Rahman (1972).
The Ahsanullah test is a statistical hypothesis test used to assess whether a given sample of data is consistent with an exponential distribution. This test is based on a characterization of the exponential distribution through properties of order statistics and related distributional identities.

Hypothesis of Exponentiality
The hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution. The exponential distribution is widely used in reliability theory, survival analysis, queuing theory, and other fields where waiting times or lifetimes are modeled.

Test Statistic
The Ahsanullah test statistic is based on comparing two empirical probabilities formed from pairs and single observations in the sample. Under exponentiality, the compared quantities are expected to be close. Large positive values of the statistic indicate departure from the exponential model in the right-tailed direction.

Calculate the Test Statistic: The test statistic counts, over all triples of observations, how often the absolute difference of two observations is smaller than a third observation and how often twice the smaller of two observations is smaller than a third observation. The normalized difference between these counts is used as the statistic.

Limitations
The statistic uses a triple summation over the sample, so the direct computation can be expensive for large sample sizes.

The test is designed around the characterization of exponentiality and should be calibrated with appropriate critical values or simulation when exact finite-sample behavior is required.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AhsanullahExponentialityGofStatistic,
)


test_statistic = AhsanullahExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Ahsanullah test is a test for the hypothesis of exponentiality based on a characterization of the exponential distribution. The test statistic is

$$ A_n = \frac{H_n - G_n}{n^3} $$

where

$$ H_n = \sum_{k=1}^{n} \sum_{i=1}^{n} \sum_{j=1}^{n} I(|X_i - X_j| < X_k) $$

and

$$ G_n = \sum_{k=1}^{n} \sum_{i=1}^{n} \sum_{j=1}^{n} I(2 min(X_i, X_j) < X_k). $$

Here $I(\cdot)$ is the indicator function and $n$ is the sample size. The implemented statistic returns the normalized difference between these two triple counts.

## Author(s)
Lev Golofastov

## References
Ahsanullah, M. and Rahman, M. (1972): A characterization of the exponential distribution. - Journal of Applied Probability, vol. 9, no. 2, pp. 457-461.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    AhsanullahExponentialityGofStatistic,
)


test_statistic = AhsanullahExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
