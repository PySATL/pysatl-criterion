# Raschke biased-transformation test for Beta

## Description

Tests the composite hypothesis of iid $\mathrm{Beta}(a,b)$ observations,
$a,b>0$, on fixed support $[0,1]$. Both shapes are unknown. Sections 4–5
of Raschke (2009) prescribe Beta maximum likelihood, transformation to normal
quantiles, normal maximum likelihood and corrected Anderson–Darling.

## Usage

```python
from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit import RaschkeBetaGofStatistic
from pysatl_criterion.hypothesis_testing.beta_bootstrap import parametric_bootstrap_beta

statistic = RaschkeBetaGofStatistic(Beta.DEFAULT.parse({}))
sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
value = statistic.execute_statistic(sample)
result = parametric_bootstrap_beta(statistic, sample, n_resamples=1999, random_state=2026)
```

## Arguments and statistic

`parameters` must leave both shapes unknown. `rvs` must contain at least three
finite real observations strictly in $(0,1)$ and be nonconstant. Endpoints,
masked data, and out-of-support observations raise `ValueError`. MLE failures
propagate. Shapes are fitted with fixed location 0 and scale 1 on every call.
The two-observation case is excluded because standardizing two distinct normal
scores always produces $(-1,1)$ and hence a constant statistic.

Let $F_{a,b}$ be the Beta CDF, $\Phi$ the standard normal CDF, and
$(\hat a,\hat b)$ the Beta MLE. Set

$$
y_i=\Phi^{-1}(F_{\hat a,\hat b}(x_i)),\quad
\hat\mu=\frac1n\sum_i y_i,\quad
\hat\sigma^2=\frac1n\sum_i(y_i-\hat\mu)^2,\quad
z_i=(y_i-\hat\mu)/\hat\sigma.
$$

The normal scale uses the MLE divisor $n$. For ordered $z_{(i)}$, compute

$$
A^2=-n-\frac1n\sum_{i=1}^n(2i-1)
\{\log\Phi(z_{(i)})+\log[1-\Phi(z_{(n+1-i)})]\},
\qquad A^{*2}=A^2(1+0.75/n+2.25/n^2).
$$

The result is $A^{*2}$, a scalar with right critical region. Identifiers are
`RASCHKE` and `RASCHKE_BETA_GOODNESS_OF_FIT`.

## Calibration and limitations

The original paper transfers fitted-normal critical values approximately,
with simulation evidence for selected shapes and sample sizes. This does not
establish a parameter-free Beta null law or universal omnibus consistency.
The transformed observations share estimated parameters and are not an iid
normal sample. Finite-sample calibration must account for the Beta shapes;
no universal limiting law is used here.

The supplied bootstrap is an explicitly different calibration from the paper's
normal tables: simulate from the observed Beta MLE, then repeat **both fits**,
the transformation, and the AD correction for every replicate. It returns
an approximate p-value $(1+\#\{T_b^*\ge T\})/(B+1)$ and empirical upper
critical quantile (`inverted_cdf`). `rejected` follows the strict critical-value
comparison; the corrected p-value decision can differ for finite $B$.
`significance_level` is in $(0,1)$, `n_resamples=B` is a positive integer, and
an integer `random_state` ensures reproducibility. Bootstrap validity for
this transformed test is not established by the Ebner–Liebenberg theorem;
no exact finite-sample level or blanket consistency claim is made.

Both Beta tails are evaluated to avoid subtracting a CDF rounded to 1.
Normal log-CDF/log-SF keep the AD sum stable. Unrepresentable transforms or
failed fits raise errors, including in bootstrap; no clipping or selective
removal of failed replicates occurs. Very small shape parameters can yield
machine-rounded endpoints in simulated samples.

## Implementation

[RaschkeBetaGofStatistic](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/beta.py)
reuses the shared `ADStatistic` kernel.
[Bootstrap calibration](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/hypothesis_testing/beta_bootstrap.py)
lives in the hypothesis-testing layer and requires the observed sample.

## References

Raschke, M. (2009). *The Biased Transformation and Its Application in
Goodness-of-Fit Tests for the Beta and Gamma Distribution.* Communications
in Statistics – Simulation and Computation, 38, 1870–1890.
DOI: [10.1080/03610910903152631](https://doi.org/10.1080/03610910903152631).
[Author manuscript, Sections 4–5](https://www.researchgate.net/publication/220504959_The_Biased_Transformation_and_Its_Application_in_Goodness-of-Fit_Tests_for_the_Beta_and_Gamma_Distribution).

## Stored calibration

`StorageLimitDistributionResolver` rejects this composite hypothesis because
the storage key lacks fitted shapes. Use `parametric_bootstrap_beta` with the
observed sample; a stored law indexed only by code and sample size is invalid.
