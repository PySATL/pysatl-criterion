# Stein test for uniformity

## Description
Performs a Stein-type U-statistic test for the hypothesis of uniformity on the interval $[a, b]$.
The implementation standardizes observations to $[0, 1]$ and computes a pairwise kernel statistic.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    SteinUniformGofStatistic,
)


test_statistic = SteinUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}))
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

The returned statistic is the signed average of this kernel over unordered pairs. At least two observations are required. Both positive and negative departures matter, so the alternative is two-sided.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Sreedevi, E.P. and Kattumannil, S.K. (2023): [Goodness of fit test for uniform distribution with censored observation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9869324/). Journal of the Korean Statistical Society 52, 382–394.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import (
    SteinUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = SteinUniformGofStatistic(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
