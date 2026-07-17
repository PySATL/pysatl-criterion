# Stein test for uniformity

## Description
Performs a Stein-type U-statistic test for the hypothesis of uniformity on the interval $[a, b]$.
The implementation standardizes observations to $[0, 1]$ and computes a pairwise kernel statistic.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SteinUniformGofStatistic,
)


test_statistic = SteinUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For standardized observations, the implementation computes a U-statistic with pairwise kernel

$$ h(x, y) = \frac{1}{2}\left(2\max(x, y) - 2x - 2y + x^2 + y^2\right). $$

The returned statistic is the average of this kernel over unordered pairs.

## Author(s)
Alexey Mironov

## References
Stein, C. (1972): A bound for the error in the normal approximation to the distribution of a sum of dependent random variables. - Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability, vol. 2, pp. 583-602.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    SteinUniformGofStatistic,
)


test_statistic = SteinUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
