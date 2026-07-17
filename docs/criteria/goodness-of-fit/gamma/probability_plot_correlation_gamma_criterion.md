# Probability plot correlation test for gamma distribution

## Description
Performs probability plot correlation coefficient goodness-of-fit test for the hypothesis that the sample comes from a gamma distribution.
The statistic measures linear alignment between ordered sample values and theoretical gamma quantiles.

Hypothesis of Gamma Distribution
The null hypothesis is that the data comes from a gamma distribution with positive shape parameter `alpha` and positive rate parameter `beta`.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ProbabilityPlotCorrelationGammaGofStatistic,
)


test_statistic = ProbabilityPlotCorrelationGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```

## Arguments
`alpha` - positive shape parameter of the gamma distribution. Default value is `1.0`.

`beta` - positive rate parameter of the gamma distribution. Default value is `1.0`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
The implementation computes plotting positions, evaluates theoretical gamma quantiles, and returns one minus the correlation coefficient between the ordered sample and the expected quantiles.
Values near zero indicate stronger linear alignment with the gamma model.

## Author(s)
Alexey Mironov

## References
Filliben, J.J. (1975): The probability plot correlation coefficient test for normality. - Technometrics, vol. 17, pp. 111-117.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ProbabilityPlotCorrelationGammaGofStatistic,
)


test_statistic = ProbabilityPlotCorrelationGammaGofStatistic(alpha=2, beta=1)
statistic_result = test_statistic.execute_statistic([0.42, 0.77, 1.05, 1.48, 1.96, 2.34, 3.12])
print(statistic_result)
```
