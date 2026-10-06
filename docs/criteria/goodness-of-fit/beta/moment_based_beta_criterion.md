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
from pysatl_criterion.statistics.goodness_of_fit import (
    MomentBasedBetaGofStatistic,
)


test_statistic = MomentBasedBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

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
from pysatl_criterion.statistics.goodness_of_fit import (
    MomentBasedBetaGofStatistic,
)


test_statistic = MomentBasedBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
