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

The hypothesis-testing layer implements this procedure with reproducible RNG:

```python
from pysatl_criterion.hypothesis_testing.beta_bootstrap import parametric_bootstrap_beta

result = parametric_bootstrap_beta(
    statistic, sample, significance_level=0.05, n_resamples=1999, random_state=2026
)
print(result.statistic, result.critical_value, result.p_value, result.rejected)
```

The function uses the inverse empirical CDF for the upper critical quantile.
`rejected` means strictly greater than that quantile, following the paper.
The reported p-value is $(1+\#\{T_b^*\ge T_n\})/(B+1)$, where $B$ is
`n_resamples`; its decision can differ from the critical-value decision at
finite $B$. Every replicate is refitted. Fit failures propagate; simulated
endpoints caused by floating-point rounding are neither clipped nor dropped.

The p-value is a bootstrap approximation; the correction does not make a
composite-null bootstrap exact in finite samples. More replicates reduce
simulation error. Universal critical values or arbitrary Beta(1, 1)
calibration are inappropriate. An empty composite hypothesis alone does not
specify the generating shapes; calibration needs the observed sample.

## Null limit, consistency and implementation

The null limit is $\sum_{j\ge1}\lambda_j(a,b)Z_j^2$, with independent
standard normal $Z_j$ and eigenvalues of the fitted-process covariance
operator (Corollary 2.2). The weights depend on both shapes. Section 3 proves
consistency against alternatives for which the shape estimators converge to
finite positive limits, as assumed in equation (6). This is an omnibus
characterization under those conditions, not an unconditional assertion for
all possible samples or endpoint-contaminated distributions.

[Statistic implementation](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/beta.py)
and [bootstrap implementation](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/hypothesis_testing/beta_bootstrap.py).
The full identifier is `EL_BETA_GOODNESS_OF_FIT`.

## References

B. Ebner and S. C. Liebenberg,
[On a new test of fit to the beta distribution](https://arxiv.org/pdf/2009.13995)
(2020 preprint), Corollary 1.2, equation (3), and Section 2.

Ebner, B., Liebenberg, S. C. (2021). *On a new test of fit to the beta distribution.*
Stat, 10, e341. DOI: [10.1002/sta4.341](https://doi.org/10.1002/sta4.341).

## Stored calibration

`StorageLimitDistributionResolver` rejects this composite hypothesis because
the storage key lacks fitted shapes. Use `parametric_bootstrap_beta` with the
observed sample; a stored law indexed only by code and sample size is invalid.
