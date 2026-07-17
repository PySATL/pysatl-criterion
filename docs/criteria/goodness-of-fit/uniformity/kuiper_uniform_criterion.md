# Kuiper test for uniformity

## Description
Performs the Kuiper goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic combines the largest positive and negative deviations between the empirical and theoretical distribution functions.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperUniformGofStatistic,
)


test_statistic = KuiperUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
After standardizing observations to $[0, 1]$, the implementation computes

$$ V = D^+ + D^- $$

where

$$ D^+ = \max_i\left(\frac{i}{n} - U_{(i)}\right), \quad D^- = \max_i\left(U_{(i)} - \frac{i - 1}{n}\right). $$

Large values indicate stronger deviation from the uniform model.

## Author(s)
Alexey Mironov

## References
Kuiper, N.H. (1960): Tests concerning random points on a circle. - Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A, vol. 63, pp. 38-47.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperUniformGofStatistic,
)


test_statistic = KuiperUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
