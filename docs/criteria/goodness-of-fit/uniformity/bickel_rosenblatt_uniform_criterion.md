# Bickel-Rosenblatt test for uniformity

## Description
Computes a Bickel-Rosenblatt-type density-based goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic compares a kernel density estimate of the standardized sample with the unit uniform density.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    BickelRosenblattUniformGofStatistic,
)


test_statistic = BickelRosenblattUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}), bandwidth='auto')
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`bandwidth` - kernel bandwidth or `auto`. Default value is `auto`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes observations to $[0, 1]$, uses a Gaussian kernel density estimate, and analytically evaluates

$$ \int_0^1 \left(\hat f(x) - 1\right)^2 dx. $$

The bandwidth is measured in unit-interval coordinates. `auto` uses $h=1.06s_n n^{-1/5}$, with population standard deviation $s_n$. A constant sample requires an explicit positive bandwidth. No asymptotic centering, boundary correction, or $n\sqrt h$ scaling is applied. Calibration must use this exact statistic and bandwidth rule. Large values indicate stronger deviation from the uniform density.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Bickel, P.J. and Rosenblatt, M. (1973): On some global measures of the deviations of density function estimates. - Annals of Statistics, vol. 1, pp. 1071-1095.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

Algorithm options (`bandwidth` here) are passed separately, outside `ParameterValues`.

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import (
    BickelRosenblattUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = BickelRosenblattUniformGofStatistic(parameters, bandwidth="auto")
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
