# Cramer-von Mises test for uniformity

## Description

`CrammerVonMisesUniformGofStatistic` has algorithm code
`CVM_UNIFORM_GOODNESS_OF_FIT`. See the [uniformity catalog](index.md).

Performs the Cramer-von Mises goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic accumulates squared differences between the empirical distribution function and the theoretical uniform cumulative distribution function.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

Test Statistic
The observations are sorted, transformed by the reference uniform cumulative distribution function, and compared with expected uniform plotting positions.

## Usage
```python
from pysatl_criterion.distribution import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    CrammerVonMisesUniformGofStatistic,
)


test_statistic = CrammerVonMisesUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}))
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - explicitly fixed left boundary of the uniform distribution.

`b` - explicitly fixed right boundary of the uniform distribution.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The Cramer-von Mises statistic is

$$ W_n^2 = \frac{1}{12n} + \sum_{i=1}^{n} \left(F_0(X_{(i)}) - \frac{2i - 1}{2n}\right)^2 $$

where $X_{(i)}$ is the $i$-th order statistic and $F_0$ is the reference uniform cumulative distribution function.
Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References

Cramer, H. (1928): On the composition of elementary errors. *Skandinavisk Aktuarietidskrift*, 11, 141–180.

von Mises, R. (1931): *Wahrscheinlichkeitsrechnung und ihre Anwendung in der Statistik und theoretischen Physik*. Leipzig: Deuticke.

Zhang, J. (2002). [Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337). *Journal of the Royal Statistical Society: Series B*, 64(2), 281–294.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

```python
from pysatl_criterion.distribution import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit.continuous import (
    CrammerVonMisesUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = CrammerVonMisesUniformGofStatistic(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
