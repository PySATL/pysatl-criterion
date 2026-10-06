# Censored Stein test for uniformity

## Description
Performs a censored Stein-type U-statistic test for the hypothesis of uniformity on the interval $[a, b]$.
When no censoring is supplied, the implementation falls back to `SteinUniformGofStatistic`.

Hypothesis of Uniformity
The null hypothesis concerns latent lifetimes $X\sim U(a,b)$, observed as $Y=\min(X,C)$ under independent right censoring.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    CensoredSteinUniformGofStatistic,
)


test_statistic = CensoredSteinUniformGofStatistic(a=0, b=1)
statistic_result = test_statistic.execute_statistic(
    [0.12, 0.25, 0.41, 0.53, 0.77, 0.91],
    censoring_indices=[0, 0, 0, 0, 0, 0],
)
print(statistic_result)
```

## Arguments
`a` - left boundary of the uniform distribution. Default value is `0`.

`b` - right boundary of the uniform distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

`censoring_indices` - binary array where `1` indicates a censored observation and `0` indicates an uncensored observation.

## Details
The implementation standardizes observations to $[0, 1]$.
For censored data, the reverse Kaplan–Meier estimator supplies $\widehat K_c(Y_i-)$, the left limit of censoring survival. Observed events precede censorings at tied times. With $c_i=1$ for censoring, $w_i=(1-c_i)/\widehat K_c(Y_i-)$:

$$ \widehat\Delta_c=\frac{2}{n(n-1)}\sum_{i<j}w_iw_jh(U_i,U_j). $$

The denominator uses the full sample size, including censored observations. At least two observations are required. The returned statistic is signed and uses a two-sided alternative. Fewer than two uncensored observations give an empty sum (zero), which alone provides no evidence of fit. Censored-data calibration must reproduce the censoring mechanism; the generic complete-data Monte Carlo resolver is insufficient.

## Author(s)
Aleksandr Podmarev, Alexey Mironov

## References
Sreedevi, E.P. and Kattumannil, S.K. (2023): [Goodness of fit test for uniform distribution with censored observation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9869324/). Journal of the Korean Statistical Society 52, 382–394.

## Examples

### Constructing with `from_parameters`

Create `ParameterValues` with both fixed boundaries, then pass it to
`from_parameters`. The method validates the supported parameterization and fixed
parameter set before calling the constructor. Omitted bounds are not filled in:
a partial mapping such as `{"a": 0}` raises `ValueError` in `from_parameters`.

```python
from pysatl_criterion.distribution.distributions import UniformDistributionDescriptor as Uniform
from pysatl_criterion.statistics.goodness_of_fit import (
    CensoredSteinUniformGofStatistic,
)


parameters = Uniform.DEFAULT.parse({"a": 0, "b": 1})
test_statistic = CensoredSteinUniformGofStatistic.from_parameters(parameters)
assert test_statistic.hypothesis().parameters() == {"a": 0, "b": 1}
statistic_result = test_statistic.execute_statistic(
    [0.12, 0.25, 0.41, 0.53, 0.77, 0.91],
    censoring_indices=[0, 0, 0, 0, 0, 0],
)
print(statistic_result)
```
