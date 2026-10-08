# Moment-based test for beta distribution

## Description

No scientific source for this exact Beta formula was identified. This class is
a locally defined discrepancy, not a claimed named published criterion.

Performs moment-based goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic compares sample mean and sample variance with the theoretical beta distribution moments.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    MomentBasedBetaGofStatistic,
)


test_statistic = MomentBasedBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`parameters` - `BetaDistributionDescriptor.DEFAULT.parse({"a": alpha, "b": beta})`. Both shapes must be explicitly supplied, positive and finite.

`alpha` and `beta` are fixed independently of the sample. Neither is estimated or filled in by the constructor.

`rvs` - a one-dimensional sample of at least two observations of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
For a beta distribution,

$$ E[X] = \frac{\alpha}{\alpha + \beta} $$

and

$$ Var(X) = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}. $$

For $d=(\bar X-\mu,S^2-\sigma^2)$ with unbiased sample variance, the statistic is

$$ T=n d^T\Sigma^{-1}d,\qquad
\Sigma=\begin{pmatrix}\sigma^2&\mu_3\\\mu_3&\mu_4-\sigma^4\end{pmatrix}. $$

Here $\mu_r$ denotes the theoretical central moment. At least two observations are required.
The limiting null law is chi-squared with two degrees of freedom for fixed positive shapes;
finite-sample calibration still depends on both shapes and sample size.

This statistic compares selected distribution functionals and need not detect every
non-Beta alternative. Previously generated critical values must be recalculated
when the statistic or its estimator settings change.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## Examples
```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit import (
    MomentBasedBetaGofStatistic,
)


test_statistic = MomentBasedBetaGofStatistic(BetaDistributionDescriptor.DEFAULT.parse({'a': 2, 'b': 5}))
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Omnibus limitation

Matching the mean and variance does not characterize the Beta family. This
criterion is not an omnibus GOF test and is not a replacement for EDF tests
or Ebner–Liebenberg. It is retained as a moment discrepancy.


## Numerical evaluation

For $s=\alpha+\beta$, standardized skewness $g$, $d_1=(\bar X-\mu)/\sigma$
and $d_2=S^2/\sigma^2-1$, the exact Schur complement is

$$r=E[((X-\mu)/\sigma)^4]-1-g^2=\frac{2s}{s+3}(1+g^2/4).$$

The equivalent statistic $T=n[d_1^2+(d_2-gd_1)^2/r]$ avoids subtracting
nearly equal fourth moments. Unrepresentable results raise `ValueError`.
The mathematical statistic and its asymptotic interpretation are unchanged.
