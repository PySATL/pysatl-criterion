# Ebner–Liebenberg test for the Beta distribution

## Description

The criterion tests the composite null hypothesis

$$
H_0: X_1,\ldots,X_n \overset{\mathrm{iid}}{\sim}
\operatorname{Beta}(a,b), \qquad a,b>0,
$$

against general alternatives on the fixed support $[0,1]$. Both shape
parameters are unknown and fitted by maximum likelihood on each sample;
location and scale remain fixed at 0 and 1.

The test is based on the conditional-moment characterization of the Beta
family in Ebner and Liebenberg. Large values of the statistic provide evidence
against the null hypothesis. This is the new statistic proposed in the paper,
not the fitted Kolmogorov–Smirnov statistic used there as a competitor.

## Usage

```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit import EbnerLiebenbergBetaGofStatistic

statistic = EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))
sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
value = statistic.execute_statistic(sample)
print(value)
```

## Arguments and return value

| Argument | Meaning |
| --- | --- |
| `parameters` | `Beta.DEFAULT.parse({})`: neither shape is fixed. Supplying either shape is unsupported. |
| `rvs` | One-dimensional sample of at least two finite real observations strictly inside `(0, 1)`. The sample must be nonconstant. |
| `**kwargs` | Reserved for interface compatibility; ignored. |

`execute_statistic` returns a nonnegative Python `float`. The class computes
only the statistic, without a p-value or a hypothesis decision. Fitted shapes
are local to each call and do not change `hypothesis().parameters()`, which
remains `{}`. The input sample is not modified.

Invalid samples, including endpoints, masked observations and constant
samples, raise `ValueError`. Failures of the maximum-likelihood solver
propagate to the caller. Nonfinite numerical results or materially negative
residuals also raise `ValueError`; negative residuals within the implementation's
roundoff tolerance are set to zero.

The short identifier is `EL`. `alternative()` returns `RightAlternative`.

## Test statistic

Let $\hat a,\hat b$ be the maximum-likelihood estimates and define
$c_j=(\hat a+\hat b)x_j-\hat a$. The integral definition is

$$
T_n=n\int_0^1\left[
\frac1n\sum_{j=1}^n c_j\mathbf1\{x_j\ge t\}
-\frac{t^{\hat a}(1-t)^{\hat b}}{B(\hat a,\hat b)}
\right]^2dt.
$$

Here $B$ is the beta function. Equation (3) of the source gives

$$
\begin{aligned}
T_n={}&\frac1n\sum_{j,k=1}^n c_jc_k\min(x_j,x_k)\\
&-2\frac{B(\hat a+1,\hat b+1)}{B(\hat a,\hat b)}
\sum_{j=1}^n c_j F_{\hat a+1,\hat b+1}(x_j)\\
&+n\frac{B(2\hat a+1,2\hat b+1)}{B(\hat a,\hat b)^2},
\end{aligned}
$$

where $F_{a,b}$ is the Beta CDF.

For sorted observations, set $x_{(0)}=0$ and
$S_i=\sum_{j=i}^n c_{(j)}$. The implementation evaluates the first term as

$$
\frac1n\sum_{i=1}^n (x_{(i)}-x_{(i-1)})S_i^2.
$$

This avoids an $n\times n$ matrix. After parameter fitting, evaluating the
statistic requires $O(n\log n)$ time and $O(n)$ memory. The last beta-function
ratio uses logarithms, and the cross-term ratio uses the identity
$B(a+1,b+1)/B(a,b)=ab/[(a+b)(a+b+1)]$.

## Calibration

The null distribution depends on the unknown shapes. Section 2 of the paper
uses a parametric bootstrap:

1. Fit both shapes to the observed sample and calculate its statistic.
2. Generate bootstrap samples of the same size from the fitted Beta law.
3. Calculate the statistic on every bootstrap sample, **refitting both shapes
   each time**.
4. Reject when the observed statistic exceeds the empirical upper critical
   quantile at the chosen significance level.

The following standalone example performs that calibration explicitly;
bootstrap simulation is not a method of the statistic class.

```python
import numpy as np
from scipy import stats

from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit import EbnerLiebenbergBetaGofStatistic

sample = np.array([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
statistic = EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))
observed = statistic.execute_statistic(sample)
a_hat, b_hat, _, _ = stats.beta.fit(sample, floc=0, fscale=1)

rng = np.random.default_rng(2026)
replicates = 1999
significance = 0.05
bootstrap = np.empty(replicates)
for i in range(replicates):
    simulated = rng.beta(a_hat, b_hat, size=sample.size)
    bootstrap[i] = statistic.execute_statistic(simulated)

critical_value = np.quantile(bootstrap, 1 - significance)
reject = observed > critical_value
# Monte Carlo estimate with a finite-simulation correction.
p_value = (1 + np.count_nonzero(bootstrap >= observed)) / (replicates + 1)
print(observed, critical_value, p_value, reject)
```

The p-value is a bootstrap approximation; the correction does not make a
composite-null bootstrap exact in finite samples. More replicates reduce
simulation error. Universal critical values or arbitrary Beta(1, 1)
calibration are inappropriate. An empty composite hypothesis alone does not
specify the generating shapes; calibration needs the observed sample.

## References

B. Ebner and S. C. Liebenberg,
[On a new test of fit to the beta distribution](https://arxiv.org/pdf/2009.13995)
(2020 preprint), Corollary 1.2, equation (3), and Section 2.
