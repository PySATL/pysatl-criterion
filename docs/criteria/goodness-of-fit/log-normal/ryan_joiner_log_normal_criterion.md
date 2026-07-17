# Ryan-Joiner test for log-normal distribution

## Description
Performs Ryan-Joiner goodness-of-fit test for the hypothesis that the sample comes from a log-normal distribution.
The implementation uses positive observations and either evaluates the log-normal model directly or applies a normality statistic to standardized logarithms.

Hypothesis of Log-Normal Distribution
The null hypothesis is that the data comes from a log-normal distribution with positive shape parameter `s` and positive scale parameter `scale`.
Observations passed to `execute_statistic` should be positive; tests that receive non-positive observations may return `inf`.

Test Statistic
The statistic is based on the wrapped Ryan-Joiner normality statistic applied to standardized log observations.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RyanJoinerLogNormalGofStatistic,
)


test_statistic = RyanJoinerLogNormalGofStatistic(s=1, scale=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.58, 0.76, 0.93, 1.12, 1.35, 1.61, 1.92, 2.28, 2.71, 3.23, 3.84])
print(statistic_result)
```

## Arguments
`s` - positive shape parameter of the log-normal distribution. Default value is `1`.

`scale` - positive scale parameter of the log-normal distribution. Default value is `1`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For dynamically generated log-normal tests, the implementation transforms observations with

$$ Z_i = \frac{\log X_i - \log(scale)}{s} $$

and applies the corresponding normality statistic to the standardized values. Explicit EDF and KL-based implementations use the formulas in `pysatl_criterion.statistics.goodness_of_fit.log_normal`.

## Author(s)
Anton Dorosev, Alexey Mironov

## References
The statistic follows the implementation in `pysatl_criterion.statistics.goodness_of_fit.log_normal`.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    RyanJoinerLogNormalGofStatistic,
)


test_statistic = RyanJoinerLogNormalGofStatistic(s=1, scale=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.58, 0.76, 0.93, 1.12, 1.35, 1.61, 1.92, 2.28, 2.71, 3.23, 3.84])
print(statistic_result)
```
