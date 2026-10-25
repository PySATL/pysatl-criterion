# Kuiper test for uniformity

## Description

`KuiperUniformGofStatistic` has algorithm code
`KUIPER_UNIFORM_GOODNESS_OF_FIT`. See the [uniformity catalog](index.md).

Performs the Kuiper goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic combines the largest positive and negative deviations between the empirical and theoretical distribution functions.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.distribution import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    KuiperUniformGofStatistic,
)


test_statistic = KuiperUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}))
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - explicitly fixed left boundary of the uniform distribution.

`b` - explicitly fixed right boundary of the uniform distribution.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
After standardizing observations to $[0, 1]$, the implementation computes

$$ V = D^+ + D^- $$

where

$$ D^+ = \max_i\left(\frac{i}{n} - U_{(i)}\right), \quad D^- = \max_i\left(U_{(i)} - \frac{i - 1}{n}\right). $$

Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References

Kuiper, N. H. (1960). [Tests concerning random points on a circle](https://doi.org/10.1016/S1385-7258(60)50006-0). *Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A*, 63, 38–47.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    KuiperUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = KuiperUniformGofStatistic(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
