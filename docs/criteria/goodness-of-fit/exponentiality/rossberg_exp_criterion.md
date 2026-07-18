# Test for exponentiality based on Rossberg characterization

## Description
Performs a test for the composite hypothesis of exponentiality based on Rossberg characterization, see e.g. Rossberg (1972) and Volkova (2010).
The Rossberg test is a statistical hypothesis test used to assess the composite hypothesis of exponentiality. This test is designed to determine whether a given sample of data is consistent with an exponential distribution by comparing empirical distributions motivated by a characterization involving differences of order statistics.

Composite Hypothesis of Exponentiality
The composite hypothesis of exponentiality refers to the null hypothesis that the data comes from an exponential distribution with an unspecified rate parameter
lambda. The Rossberg statistic is scale invariant because the characterization uses comparisons between sample observations and differences of order statistics.

Test Statistic
The Rossberg test statistic is based on a characterization of the exponential distribution using order statistics. It compares the empirical distribution of the difference between the first two order statistics in triples with the empirical distribution of the minimum of pairs.

Calculate the Test Statistic: For each observation, count the proportion of sample triples for which the second order statistic minus the first order statistic is smaller than that observation, subtract the corresponding proportion of sample pairs for which the pair minimum is smaller than that observation, and average the differences over the sample.

Limitations
The test assumes nonnegative lifetime-type observations.

The implementation uses nested loops over triples and pairs, so computation can be expensive for large samples.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RossbergExponentialityGofStatistic,
)


test_statistic = RossbergExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```

## Arguments
`rvs` - array-like sample data passed to `execute_statistic`.

## Details

The Rossberg test is a test for the composite hypothesis of exponentiality. Let $F_n$ be the empirical distribution function. Define

$$ H_n(t) = {n \choose 3}^{-1}\sum_{1 \leq i<j<k \leq n} I(X_{2,\{i,j,k\}}-X_{1,\{i,j,k\}} < t), $$

where $X_{1,\{i,j,k\}}$ and $X_{2,\{i,j,k\}}$ are the first and second order statistics of the triple $X_i,X_j,X_k$. Also define

$$ G_n(t) = {n \choose 2}^{-1}\sum_{1 \leq i<j \leq n} I(\min(X_i,X_j) < t). $$

The test statistic implemented here is

$$ RS_n = \int_{0}^{\infty}(H_n(t)-G_n(t))dF_n(t). $$

In computational form this is the sample average of $H_n(X_m)-G_n(X_m)$ over $m=1,\ldots,n$.

Under exponentiality,

$$ \sqrt{n}RS_n \xrightarrow{d} N\left(0,\frac{52}{1125}\right). $$

Large values of $RS_n$ indicate larger deviations from the exponential model in the direction used by the implemented right-sided alternative.

## Author(s)
Lev Golofastov

## References
Rossberg, H.J. (1972): Characterization of the exponential and the Pareto distributions by means of some properties of the distributions which the differences and quotients of order statistics are subject to. - Mathematische Operationsforschung und Statistik, vol. 3, no. 3, pp. 207-216. https://doi.org/10.1080/02331887208801077

Volkova, K.Yu. (2010): On asymptotic efficiency of exponentiality tests based on Rossberg characterization. - Journal of Mathematical Sciences, vol. 167, no. 4, pp. 486-494.

Henze, N. and Meintanis, S.G. (2005): Recent and classical tests for exponentiality: a partial review with comparisons. - Metrika, vol. 61, pp. 29-45.

## Examples

```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RossbergExponentialityGofStatistic,
)


test_statistic = RossbergExponentialityGofStatistic()
statistic_result = test_statistic.execute_statistic([1, 2, 3, 4, 5, 6, 7])
print(statistic_result)
```
