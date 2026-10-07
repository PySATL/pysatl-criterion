# Mode test for beta distribution

## Description

No scientific source for this exact Beta formula was identified. This class is
a locally defined discrepancy, not a claimed named published criterion.

Performs mode-based goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares an estimated sample mode with the theoretical beta distribution mode.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with shape parameters `alpha > 1` and `beta > 1` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    ModeBetaGofStatistic,
)


test_statistic = ModeBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first shape parameter of the beta distribution. Must be greater than `1`. Default value is `2`.

`beta` - second shape parameter of the beta distribution. Must be greater than `1`. Default value is `2`.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
For `alpha > 1` and `beta > 1`, the beta distribution mode is

$$ \frac{\alpha - 1}{\alpha + \beta - 2}. $$

The implementation uses a Gaussian KDE with Scott bandwidth, locates candidate
maxima on a grid over the sample range whose size grows with sample size, and
refines each candidate by bounded optimization. Computation uses coordinates
scaled to [0,1] within the sample range, including both sample extrema. Gaussian
KDE modes lie within the sample range, so support points outside it cannot maximize
the density. Scaling prevents KDE variance underflow for very narrow samples. At least two observations and a nonconstant sample are required.
The returned value is `sqrt(n) * abs(estimated_mode - theoretical_mode)`.
The factor `sqrt(n)` is a scale convention, not a claim of asymptotic normality.
Calibrate by simulation for both fixed shapes and sample size, using the same KDE
procedure. Finite-sample KDE boundary bias remains; the search is numerical.

This statistic compares selected distribution functionals and need not detect every
non-Beta alternative. Previously generated critical values must be recalculated
when the statistic or its estimator settings change.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## Examples
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    ModeBetaGofStatistic,
)


test_statistic = ModeBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
