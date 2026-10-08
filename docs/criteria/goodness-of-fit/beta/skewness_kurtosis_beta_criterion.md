# Skewness-kurtosis test for beta distribution

## Description

No scientific source for this exact Beta formula was identified. This class is
a locally defined discrepancy, not a claimed named published criterion.

Performs skewness-kurtosis goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares sample skewness and kurtosis with the theoretical beta distribution values.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    SkewnessKurtosisBetaGofStatistic,
)


test_statistic = SkewnessKurtosisBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
The implementation computes sample skewness and kurtosis and compares them with the theoretical values for the beta distribution.
Let $Z=(X-\mu)/\sigma$, $g=E[Z^3]$, and $k=E[Z^4]$.
The influence functions are $p_3(Z)=Z^3-3Z-3gZ^2/2+g/2$ and
$p_4(Z)=Z^4-4gZ-2kZ^2+k$. With $\Sigma_{ij}=E[p_i(Z)p_j(Z)]$,
the statistic is $n d^T\Sigma^{-1}d$, where $d$ contains sample skewness minus $g$
and sample excess kurtosis minus $(k-3)$. Sample estimates use `bias=False`, evaluated after affine centering and rescaling
to avoid loss of precision for nearly constant samples.
The Beta covariance uses moments through order eight, including the off-diagonal term.
At least four observations and a nonconstant sample are required.
The limiting null law is chi-squared with two degrees of freedom; finite-sample
calibration depends on both shapes and sample size.

This statistic compares selected distribution functionals and need not detect every
non-Beta alternative. Previously generated critical values must be recalculated
when the statistic or its estimator settings change.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## Examples
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    SkewnessKurtosisBetaGofStatistic,
)


test_statistic = SkewnessKurtosisBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Numerical evaluation

The influence-function covariance and Schur complement use standard-library
`decimal` arithmetic with precision adapted to the shapes. This prevents
cancellation from producing negative statistics near small positive shapes.
Unrepresentable final float results raise `ValueError`. This adds computational
cost but does not change the statistic or its asymptotic interpretation.
