# Watson test for uniformity

## Description
Performs Watson's $U^2$ goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
Watson's statistic is a rotation-invariant modification of the Cramer-von Mises statistic.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonUniformGofStatistic,
)


test_statistic = WatsonUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes the ordered observations to $[0, 1]$ and computes

$$ U^2 = \frac{1}{n}\sum_{i=1}^{n}\left(U_{(i)} - \frac{i}{n} + \frac{1}{2n} - \bar U\right)^2 + \frac{1}{12n^2}. $$

Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Watson, G.S. (1961): Goodness-of-fit tests on a circle. - Biometrika, vol. 48, pp. 109-114.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    WatsonUniformGofStatistic,
)


test_statistic = WatsonUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
