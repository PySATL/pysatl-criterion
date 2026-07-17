# Min-Toshiyuki test for gamma distribution

## Description
Performs Min-Toshiyuki tail-sensitive goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic up-weights empirical distribution deviations near the tails.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MinToshiyukiGammaGofStatistic,
)


test_statistic = MinToshiyukiGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation transforms observations with the gamma cumulative distribution function and passes the resulting values to the common Min-Toshiyuki statistic implementation.

## Author(s)
Alexey Mironov

## References
The statistic follows the implementation in `pysatl_criterion.statistics.goodness_of_fit.gamma`.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    MinToshiyukiGammaGofStatistic,
)


test_statistic = MinToshiyukiGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
