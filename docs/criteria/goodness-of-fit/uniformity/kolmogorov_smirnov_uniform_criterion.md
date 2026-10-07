# Kolmogorov-Smirnov test for uniformity

## Description
Performs the Kolmogorov-Smirnov goodness-of-fit test for the hypothesis of uniformity on the interval $[a, b]$.
The statistic measures the largest discrepancy between the empirical distribution function and the cumulative distribution function of the reference uniform distribution.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.
Observations must be finite. Values outside the null interval are accepted;
the reference CDF is 0 below `a` and 1 above `b`.

Test Statistic
The statistic is based on the maximum vertical distance between the empirical cumulative distribution function and the theoretical uniform cumulative distribution function.

## Usage
```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovUniformGofStatistic,
)


test_statistic = KolmogorovSmirnovUniformGofStatistic(UniformDistributionDescriptor.DEFAULT.parse({'a': 0, 'b': 1}))
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`alternative_type` - alternative hypothesis type used by the Kolmogorov-Smirnov statistic.

`mode` - calculation mode passed to the Kolmogorov-Smirnov statistic. Default value is `auto`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For ordered observations $X_{(i)}$, the implementation evaluates

$$ F_0(X_{(i)}) = \min\left(1,\max\left(0,\frac{X_{(i)} - a}{b - a}\right)\right). $$

and passes these values to the common Kolmogorov-Smirnov statistic implementation.
Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Kolmogorov, A.N. (1933): Sulla determinazione empirica di una legge di distribuzione. - Giornale dell'Istituto Italiano degli Attuari, vol. 4, pp. 83-91.

Smirnov, N.V. (1948): Table for estimating the goodness of fit of empirical distributions. - Annals of Mathematical Statistics, vol. 19, pp. 279-281.

## Examples

### Constructing with `ParameterValues`

Create `ParameterValues` with both fixed boundaries, then pass it to
the constructor. It validates the supported parameterization and exact fixed
parameter set. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in the constructor.

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import (
    KolmogorovSmirnovUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = KolmogorovSmirnovUniformGofStatistic(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

To fix the default interval explicitly, use
`Uniform.DEFAULT.parse({}, fill_defaults=True)` before calling the constructor.
This differs from `parse({})`, which declares no fixed bounds and is unsupported
by this KS class. For unknown bounds, see the
[Lilliefors-type implementation](lilliefors_uniform_criterion.md#examples).
