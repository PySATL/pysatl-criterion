# Lilliefors-type test for uniformity

## Description
Computes a Lilliefors-type KS statistic for the uniform family with both boundaries unknown.
This implementation uses a KS distance with boundaries estimated by maximum likelihood
as the sample minimum and maximum. It is a uniform-family adaptation, not the
classical normality test or its critical-value tables.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from some uniform distribution with unknown boundaries. `hypothesis().parameters()` returns `{}`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsTestUniformGofStatistic,
)


test_statistic = LillieforsTestUniformGofStatistic()
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
The constructor accepts no distribution parameters.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sample is sorted and transformed with

$$ \widehat F(x) = \frac{x-X_{(1)}}{X_{(n)}-X_{(1)}}. $$

The two-sided KS distance is computed against this fitted CDF. At least two distinct finite observations are required. Calibration must refit both boundaries on every simulated sample; ordinary KS tables and the normal-distribution Lilliefors tables do not apply. The reference below motivates the fitted-parameter approach, rather than uniform-specific critical values.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Lilliefors, H.W. (1967): On the Kolmogorov-Smirnov test for normality with mean and variance unknown. - Journal of the American Statistical Association, vol. 62, pp. 399-402.

## Examples

### Constructing with `from_parameters`

`from_parameters` accepts an empty `ParameterValues` object for this class.
`parse({})` leaves both boundaries unknown; it does not insert `a=0, b=1`.
The bounds are fitted only when `execute_statistic` receives the sample.

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import (
    LillieforsTestUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({})
test_statistic = LillieforsTestUniformGofStatistic.from_parameters(parameters)
assert test_statistic.hypothesis().parameters() == {}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

### Unsupported fixed boundaries

A hypothesis with either or both bounds fixed is unsupported by this class.
The following example catches the expected error:

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import LillieforsTestUniformGofStatistic

parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
try:
    LillieforsTestUniformGofStatistic.from_parameters(parameters)
except ValueError as error:
    print(error)  # Unsupported Uniform hypothesis or parameterization
```

For a specified interval, use
[`KolmogorovSmirnovUniformGofStatistic.from_parameters`](kolmogorov_smirnov_uniform_criterion.md#examples).
Do not use `fill_defaults=True` for the fitted-bound hypothesis: it would fix both
bounds and the resulting hypothesis would be rejected.
