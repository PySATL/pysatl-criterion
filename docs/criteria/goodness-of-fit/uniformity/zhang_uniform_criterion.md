# Zhang tests for uniformity

## Description
Performs Zhang's goodness-of-fit tests for the hypothesis of uniformity on the interval $[a, b]$.
The implementation supports three variants selected by `test_type`: `A`, `C`, and `K`.

Hypothesis of Uniformity
The null hypothesis is that the sample comes from a uniform distribution on the interval $[a, b]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangTestsUniformGofStatistic,
)


test_statistic = ZhangTestsUniformGofStatistic(a=0, b=1, test_type="A")
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`test_type` - Zhang statistic variant. Must be `A`, `C`, or `K`. Default value is `A`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation standardizes ordered observations to $U_{(i)} \in [0, 1]$ and computes one of the Zhang statistics using logarithmic transforms of $U_{(i)}$ and $1 - U_{(i)}$.
For $U_i=F_{a,b}(X_{(i)})$, the implemented statistics are

$$ Z_A=-\sum_{i=1}^n\left[\frac{\log U_i}{n-i+1/2}+\frac{\log(1-U_i)}{i-1/2}\right], $$

$$ Z_C=\sum_{i=1}^n\left[\log\frac{U_i^{-1}-1}{(n-1/2)/(i-3/4)-1}\right]^2, $$

$$ Z_K=\max_i\left[(i-1/2)\log\frac{i-1/2}{nU_i}+(n-i+1/2)\log\frac{n-i+1/2}{n(1-U_i)}\right]. $$

Values at or outside either support boundary give positive infinity, without arbitrary logarithm clipping. Large values indicate stronger deviation from the uniform model. Calibration must use the same `test_type`; the legacy shared `code()` does not distinguish variants.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Zhang, J. (2002): Powerful goodness-of-fit tests based on the likelihood ratio. - Journal of the Royal Statistical Society, Series B, vol. 64, pp. 281-294.

## Examples

### Constructing with `from_parameters`

Create `ParameterValues` with both fixed boundaries, then pass it to
`from_parameters`. The method validates the supported parameterization and fixed
parameter set before calling the constructor. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in `from_parameters`.

Algorithm options (`test_type` here) are passed separately, outside `ParameterValues`.

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangTestsUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = ZhangTestsUniformGofStatistic.from_parameters(parameters, test_type="A")
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
