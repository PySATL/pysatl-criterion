# Graph connected components test for gamma distribution

## Description
Performs graph connected components goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic counts connected components in a proximity graph built on gamma CDF transformed observations.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GraphConnectedComponentsGammaGofStatistic,
)


test_statistic = GraphConnectedComponentsGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The sample is transformed to gamma CDF values. A proximity graph is built on the transformed values, and the statistic is the number of connected components.

## Author(s)
Alexey Mironov

## References
The statistic follows the graph-based implementation in `pysatl_criterion.statistics.goodness_of_fit.gamma`.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    GraphConnectedComponentsGammaGofStatistic,
)


test_statistic = GraphConnectedComponentsGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
