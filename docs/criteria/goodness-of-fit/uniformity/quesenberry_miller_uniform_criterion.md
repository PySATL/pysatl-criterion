# Quesenberry-Miller test for uniformity

## Description
Performs the Quesenberry-Miller spacing test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic combines squared spacings and products of consecutive spacings.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    QuesenberryMillerUniformGofStatistic,
)


test_statistic = QuesenberryMillerUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation adds the boundaries $a$ and $b$ to the sorted sample, computes spacings $D_i$, and returns

$$ Q = \sum_i D_i^2 + \sum_i D_iD_{i+1}. $$

Large values indicate stronger deviation from the uniform spacing pattern.

## Author(s)
Alexey Mironov

## References
Quesenberry, C.P. and Miller, F.L. (1977): Power studies of some tests for uniformity. - Journal of Statistical Computation and Simulation, vol. 5, pp. 169-191.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    QuesenberryMillerUniformGofStatistic,
)


test_statistic = QuesenberryMillerUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
