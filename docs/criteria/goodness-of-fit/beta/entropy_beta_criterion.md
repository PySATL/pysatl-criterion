# Entropy test for beta distribution

## Description

No scientific source for this exact Beta formula was identified. This class is
a locally defined discrepancy, not a claimed named published criterion.

Performs entropy-based goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares a spacing-based sample entropy estimate with the theoretical entropy of the beta distribution.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    EntropyBetaGofStatistic,
)


test_statistic = EntropyBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`m` - optional window size for the spacing-based entropy estimator, passed to `execute_statistic`. The default is `min(int(sqrt(n) + 0.5), (n - 1) // 2)`. It must be an integer with `1 <= m < n/2`; at least three observations are required.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
The implementation uses a Vasicek-style spacing entropy estimator on the sorted sample.
It compares this estimate with the theoretical beta entropy computed from `betaln` and digamma terms.
A zero spacing yields entropy minus infinity and statistic plus infinity.
The default window grows while its ratio to sample size tends to zero, as required
for consistency of the Vasicek estimator.
The returned value is `sqrt(n) * abs(estimated_entropy - theoretical_entropy)`.
This scaling does not supply a universal null distribution: calibrate by simulation
for both fixed shapes, sample size, and the same window rule.

This statistic compares selected distribution functionals and need not detect every
non-Beta alternative. Previously generated critical values must be recalculated
when the statistic or its estimator settings change.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## Examples
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    EntropyBetaGofStatistic,
)


test_statistic = EntropyBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
