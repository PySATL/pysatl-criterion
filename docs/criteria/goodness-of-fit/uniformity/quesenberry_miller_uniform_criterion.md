# Quesenberry-Miller test for uniformity

## Description
Performs the Quesenberry-Miller spacing test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic combines squared spacings and products of consecutive spacings.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    QuesenberryMillerUniformGofStatistic,
)


test_statistic = QuesenberryMillerUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}))
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
Let $U_i=(X_i-a)/(b-a)$. The implementation adds boundaries 0 and 1 to the ordered transformed sample, computes unit spacings $D_i$, and returns

$$ Q = \sum_i D_i^2 + \sum_i D_iD_{i+1}. $$

Large values indicate stronger deviation from the uniform spacing pattern.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Quesenberry, C.P. and Miller, F.L. (1977): Power studies of some tests for uniformity. - Journal of Statistical Computation and Simulation, vol. 5, pp. 169-191.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import (
    QuesenberryMillerUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = QuesenberryMillerUniformGofStatistic(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
