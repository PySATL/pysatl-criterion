# Pearson chi-squared test for uniformity

## Description
Performs Pearson's chi-squared goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The implementation bins the sample and compares observed bin counts with equal expected counts.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonUniformGofStatistic,
)


test_statistic = Chi2PearsonUniformGofStatistic(a=0, b=1, bins="sturges")
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`lambda_` - power-divergence parameter passed to the common chi-squared statistic implementation. Default value is `1`.

`bins` - number of bins or bin-selection rule. Supported string values include `sturges`, `sqrt`, and `auto`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sample is divided into bins over $[a, b]$.
For $m$ bins and sample size $n$, each bin has expected count $n / m$ under the uniform null hypothesis.
The observed and expected counts are passed to the common chi-squared statistic implementation.

## Author(s)
Alexey Mironov

## References
Pearson, K. (1900): On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. - Philosophical Magazine, vol. 50, pp. 157-175.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonUniformGofStatistic,
)


test_statistic = Chi2PearsonUniformGofStatistic(a=0, b=1, bins="sturges")
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
