# Anderson-Darling test for uniformity

## Description

`AndersonDarlingUniformGofStatistic` has algorithm code
`AD_UNIFORM_GOODNESS_OF_FIT`. See the [uniformity catalog](index.md).

Performs the Anderson-Darling goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The test gives more weight to discrepancies in the tails than the Cramer-von Mises statistic.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

Test Statistic
The statistic is computed from the ordered sample using the log cumulative distribution function and log survival function of the reference uniform distribution.

## Usage
```python
from pysatl_criterion.distribution import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    AndersonDarlingUniformGofStatistic,
)


test_statistic = AndersonDarlingUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}))
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - explicitly fixed left boundary of the uniform distribution.

`b` - explicitly fixed right boundary of the uniform distribution.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For the uniform distribution on $[a, b]$,

$$ F_0(x) = \frac{x - a}{b - a}, \quad a \le x \le b. $$

The statistic uses logarithms of the uniform CDF and survival probabilities at the ordered observations. A CDF value of 0 or 1 produces positive infinity.
Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References

Anderson, T. W., and Darling, D. A. (1952). [Asymptotic Theory of Certain Goodness of Fit Criteria Based on Stochastic Processes](https://doi.org/10.1214/aoms/1177729437). *Annals of Mathematical Statistics*, 23(2), 193–212.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    AndersonDarlingUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = AndersonDarlingUniformGofStatistic(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
