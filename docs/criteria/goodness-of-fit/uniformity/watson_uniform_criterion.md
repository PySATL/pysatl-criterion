# Watson test for uniformity

## Description

`WatsonUniformGofStatistic` has algorithm code
`WATSON_UNIFORM_GOODNESS_OF_FIT`. See the [uniformity catalog](index.md).

Performs Watson's $U^2$ goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
Watson's statistic is a rotation-invariant modification of the Cramer-von Mises statistic.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.distribution import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    WatsonUniformGofStatistic,
)


test_statistic = WatsonUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}))
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - explicitly fixed left boundary of the uniform distribution.

`b` - explicitly fixed right boundary of the uniform distribution.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
With $U_{(i)}=F_{a,b}(X_{(i)})$, the implementation computes

$$ U^2 = \sum_{i=1}^{n}\left(U_{(i)} - \frac{2i-1}{2n} - \bar U + \frac12\right)^2 + \frac{1}{12n}. $$

Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References

Watson, G. S. (1961). [Goodness-of-fit tests on a circle](https://doi.org/10.1093/biomet/48.1-2.109). *Biometrika*, 48(1–2), 109–114.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    WatsonUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = WatsonUniformGofStatistic(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
