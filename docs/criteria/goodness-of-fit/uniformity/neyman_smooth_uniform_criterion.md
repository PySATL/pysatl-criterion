# Neyman smooth test for uniformity

## Description
Performs Neyman's smooth goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic uses orthonormal polynomial components on the unit interval.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    NeymanSmoothTestUniformGofStatistic,
)


test_statistic = NeymanSmoothTestUniformGofStatistic(a=0, b=1, k=4)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`k` - number of smooth components. Default value is `4`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes observations to $[0, 1]$ and computes

$$ \sum_{j=1}^{k} V_j^2, \quad V_j = \frac{1}{\sqrt n}\sum_{i=1}^{n}\phi_j(U_i). $$

The first components are implemented explicitly, and higher-order components use Legendre polynomials.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Neyman, J. (1937): Smooth test for goodness of fit. - Skandinavisk Aktuarietidskrift, vol. 20, pp. 149-199.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    NeymanSmoothTestUniformGofStatistic,
)


test_statistic = NeymanSmoothTestUniformGofStatistic(a=0, b=1, k=4)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
