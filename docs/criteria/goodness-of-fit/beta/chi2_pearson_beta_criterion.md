# Pearson chi-squared test for beta distribution

## Description
Performs Pearson's chi-squared goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The implementation compares observed histogram bin counts with expected counts under the reference beta distribution.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonBetaGofStatistic,
)


test_statistic = Chi2PearsonBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}), lambda_=1)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`lambda_` - power-divergence parameter passed to the common chi-squared statistic implementation. Default value is `1`.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
The implementation uses approximately $\sqrt n$ bins on $[0, 1]$.
Expected frequencies are computed from beta CDF differences at the bin edges.

The implementation uses `ceil(sqrt(n))` equal-width bins over `[0, 1]`.
Upper-tail probabilities use survival-function differences to avoid CDF cancellation.
Expected counts must remain representable and positive; no pooling of sparse bins
is performed. With the growing bin count, a simple chi-squared approximation is
not guaranteed. Calibrate for sample size, both shapes, and `lambda_`.
Only `lambda_=1` is Pearson; other values compute power divergence.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References

K. Pearson (1900), "On the criterion that a given system of
   deviations from the probable in the case of a correlated system of
   variables is such that it can be reasonably supposed to have arisen
   from random sampling", Philosophical Magazine 50, 157-175.
   https://doi.org/10.1080/14786440009463897

N. Cressie and T. R. C. Read (1984), "Multinomial Goodness-of-Fit
   Tests", JRSS B 46, 440-464.
   https://doi.org/10.1111/j.2517-6161.1984.tb01318.x

## Examples
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    Chi2PearsonBetaGofStatistic,
)


test_statistic = Chi2PearsonBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}), lambda_=1)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Stored calibration

Storage keys omit algorithm options. Stored calibration is allowed only for
`lambda_=1`. Other settings raise `ValueError` before storage lookup; use
`MonteCarloLimitDistributionResolver` instead. Existing stored results generated
with other options under the same key must be regenerated.
