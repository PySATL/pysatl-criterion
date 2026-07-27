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
Large values indicate stronger deviation from the uniform model.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Zhang, J. (2002): Powerful goodness-of-fit tests based on the likelihood ratio. - Journal of the Royal Statistical Society, Series B, vol. 64, pp. 281-294.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ZhangTestsUniformGofStatistic,
)


test_statistic = ZhangTestsUniformGofStatistic(a=0, b=1, test_type="A")
statistic_result = test_statistic.execute_statistic([0.12, 0.25, 0.41, 0.53, 0.77, 0.91])
print(statistic_result)
```
