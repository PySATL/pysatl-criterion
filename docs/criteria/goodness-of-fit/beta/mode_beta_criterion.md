# Mode test for beta distribution

## Description
Performs mode-based goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares an estimated sample mode with the theoretical beta distribution mode.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with shape parameters `alpha > 1` and `beta > 1` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ModeBetaGofStatistic,
)


test_statistic = ModeBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first shape parameter of the beta distribution. Must be greater than `1`. Default value is `2`.

`beta` - second shape parameter of the beta distribution. Must be greater than `1`. Default value is `2`.

`rvs` - array-like sample data passed to `execute_statistic`.

## Details
For `alpha > 1` and `beta > 1`, the beta distribution mode is

$$ \frac{\alpha - 1}{\alpha + \beta - 2}. $$

The implementation estimates the sample mode with a Gaussian kernel density estimate on a grid and returns a scaled absolute difference from the theoretical mode.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References
The statistic follows the implementation in `pysatl_criterion.statistics.goodness_of_fit.beta`.

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    ModeBetaGofStatistic,
)


test_statistic = ModeBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
