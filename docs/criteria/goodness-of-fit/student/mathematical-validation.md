# Mathematical verification of Student statistics

This report covers `continuous/student/student_statistics.py`: **21 concrete classes**, comprising
13 generated EDF criteria, 5 generated frequency criteria and 3 specialized
criteria. Lequesne has two hypothesis modes in one class. See the
[complete catalog](index.md) for class names, short codes and individual pages.

## Hypothesis and parameter estimation

The one-dimensional model is

$$
F_{\nu,\mu,s}(x)=F_{t_\nu}((x-\mu)/s),\qquad \nu>0,\quad s>0.
$$

`scale` is $s$, not variance. Where it exists, variance is
$s^2\nu/(\nu-2)$. Every class declares its own supported fixed-parameter sets;
`AbstractStudentGofStatistic` supplies no family-wide hypothesis declaration.

| Implementations | Hypothesis | Estimation performed on each sample |
|---|---|---|
| 13 EDF and 5 frequency | `df`, `loc`, `scale` all fixed | None |
| Meintanis ECF | Only `df > 2` fixed | Mean and scatter $(\nu-2)S_n/\nu$, with $S_n=n^{-1}\sum_i(X_i-\bar X)^2$ |
| Çadirci Rényi | Only `df > 2` fixed | Scatter $(\nu-2)\widehat\Sigma^2/\nu$, with unbiased sample variance (denominator $n-1$) |
| Lequesne T1 | Only `df > 2` fixed | Same unbiased variance choice as Çadirci |
| Lequesne T2 | Empty parameter set | Numerical MLE of `df`, location and scale; fitted `df` must exceed 2 |

Omitted parameters remain unknown. Fully specified parameters are not accepted
by the three specialized criteria. These implementations handle independent
one-dimensional data; they do not fit multivariate models or GARCH innovations.

## Formula checks: generated criteria

For EDF criteria, $u_i=F_{\nu,\mu,s}(X_{(i)})$, $r_i=i-1/2$,
$D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$.
For count criteria, $O_j$ denotes observed counts and $E_j=np_j>0$ expected counts.
The following table enumerates every generated criterion; definitions were
compared with its selected shared kernel and the numerical references below.

| Criterion | Implemented definition | Scientific source |
|---|---|---|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_student_criterion.md) | $D^+=\max_{1\le i\le n}(i/n-u_i)$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.1. |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_student_criterion.md) | $D^-=\max_{1\le i\le n}(u_i-(i-1)/n)$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.1. |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_student_criterion.md) | $D=\max(D^+,D^-),\quad D^+=\max_i(i/n-u_i),\quad D^-=\max_i(u_i-(i-1)/n)$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.1. |
| [Anderson–Darling](anderson_darling_student_criterion.md) | $A^2=-n-\frac1n\sum_{i=1}^n(2i-1)[\log u_i+\log(1-u_{n+1-i})]$ | [Anderson and Darling (1952), Asymptotic Theory of Certain Goodness of Fit Criteria Based on Stochastic Processes](https://doi.org/10.1214/aoms/1177729437); [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.2. |
| [Cramér–von Mises](cvm_student_criterion.md) | $W^2=\frac{1}{12n}+\sum_{i=1}^n\left(u_i-\frac{i-1/2}{n}\right)^2$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.3. |
| [Liao–Shimokawa](liao_shimokawa_student_criterion.md) | $L=\frac1{\sqrt n}\sum_{i=1}^n\frac{\max(i/n-u_i,\ u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}$ | [Liao and Shimokawa (1999), A new goodness-of-fit test for Type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965). |
| [Watson](watson_student_criterion.md) | $U^2=W^2-n(\bar u-1/2)^2,\quad W^2=\frac1{12n}+\sum_i[u_i-(i-1/2)/n]^2$ | [Watson (1961), Goodness-of-fit tests on a circle](https://doi.org/10.1093/biomet/48.1-2.109). |
| [Kuiper](kuiper_student_criterion.md) | $V=D^++D^-,\quad D^+=\max_i(i/n-u_i),\quad D^-=\max_i(u_i-(i-1)/n)$ | [Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X), equations (1)–(3). |
| [Scaled Kuiper](kuiper_scaled_student_criterion.md) | $V_s=\sqrt n(D^++D^-)$ | [Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X), equations (1)–(3). |
| [Corrected Kuiper](kuiper_corrected_student_criterion.md) | $V_c=\left(\sqrt n+0.155+\frac{0.24}{\sqrt n}\right)(D^++D^-)$ | [Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X), equations (1)–(3). |
| [Zhang Z_K](zhang_k_student_criterion.md) | $Z_K=\max_i\left[r_i\log\frac{r_i}{nu_i}+(n-r_i)\log\frac{n-r_i}{n(1-u_i)}\right],\quad r_i=i-\tfrac12$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.1). |
| [Zhang Z_A](zhang_za_student_criterion.md) | $Z_A=-\sum_{i=1}^n\left[\frac{\log u_i}{n-i+1/2}+\frac{\log(1-u_i)}{i-1/2}\right]$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.2). |
| [Zhang Z_C](zhang_zc_student_criterion.md) | $Z_C=\sum_{i=1}^n\left[\log\frac{1-u_i}{u_i}-\log\frac{n-i+1/4}{i-3/4}\right]^2$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.3). |
| [Pearson chi-squared](chi_square_student_criterion.md) | $X^2=\sum_j\frac{(O_j-E_j)^2}{E_j}$ | [Pearson (1900), On the criterion that a given system of deviations from the probable …](https://doi.org/10.1080/14786440009463897); [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x). |
| [Freeman–Tukey](freeman_tukey_student_criterion.md) | $FT=4\sum_j(\sqrt{O_j}-\sqrt{E_j})^2$ | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), power-divergence parameter $\lambda=-1/2$. |
| [Modified log-likelihood](modified_log_likelihood_student_criterion.md) | $MLL=2\sum_j E_j\log(E_j/O_j)$ | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=-1$. |
| [Neyman chi-squared](neyman_chi2_student_criterion.md) | $N=\sum_j\frac{(O_j-E_j)^2}{O_j}$ | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=-2$. |
| [Cressie–Read](cressie_read_student_criterion.md) | $CR=\frac95\sum_j O_j\left[(O_j/E_j)^{2/3}-1\right]$ | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=2/3$. |

The five count statistics correspond to power-divergence parameters
$\lambda=1,-1/2,-1,-2,2/3$. At an empty observed cell with $E_j>0$, the
contributions are respectively $E_j$, $4E_j$, infinity, infinity and zero.
The default binning is `EqualProbabilityBins()` with a sample-size-dependent
count; explicit counts and other strategies are constructor settings.

## Specialized formula checks

### Meintanis ECF

The [criterion page](meintanis_student_criterion.md) specifies the one-dimensional
specialization of [Meintanis et al. (2024)](https://doi.org/10.1111/jtsa.12713),
equations (2.3)–(2.8), (3.3) and (4.7). Its i.i.d. experiments use moment
estimation. The standardized observations are
$Y_i=(X_i-\bar X)/\sqrt{(\nu-2)S_n}$, so their target distribution has Student
scale $1/\sqrt\nu$, not unit scale. Its CF is

$$
\phi_\nu(t)=\frac{2^{1-\nu/2}}{\Gamma(\nu/2)}
|t|^{\nu/2}K_{\nu/2}(|t|),\qquad \phi_\nu(0)=1.
$$

The returned value is

$$
\frac{n}{2\pi}\int_{\mathbb R}
\left|n^{-1}\sum_i e^{itY_i}-\phi_\nu(t)\right|^2\phi_{\nu_w}(t)\,dt.
$$

`weight_df=None` selects $\nu_w=\nu$; a different positive value selects the
alternative Student weight. The change of variable used for numerical
integration retains the Jacobian, $n$ factor, both frequency half-lines and
$1/(2\pi)$ normalization. A failed quadrature raises an error.

An independent closed-form Fourier-inversion check at $\nu=3$ uses
$\phi_3(t)=(1+|t|)e^{-|t|}$, with either the same weight or a Cauchy weight.
Separate density integrals check the standard-scale CF at several degrees of
freedom, including the large-order Bessel fallback. These checks verify the
statistic's normalization, not finite-sample significance levels.

### Çadirci Rényi kNN

The [criterion page](cadirci_renyi_student_criterion.md) follows
[Çadirci et al. (2025), sections 2.1, 3, 4 and 5.2](https://arxiv.org/abs/2502.08654),
with dimension one and $q=1-2/(\nu+1)$. With
$\widehat s^2=(1-2/\nu)\widehat\Sigma^2$, equation (20) gives
$W=H_q(t_\nu(\widehat s))-\widehat H_{n,k,q}$, where

$$
\widehat H_{n,k,q}=\frac1{1-q}\log\left[
\frac1n\sum_i\{2(n-1)C_k\rho_i^{(k)}\}^{1-q}\right],\qquad
C_k=\left[\frac{\Gamma(k)}{\Gamma(k+1-q)}\right]^{1/(1-q)}.
$$

The unbiased variance, exclusion of each point from its own neighbour set,
unit-ball volume 2, factor $n-1$, and gamma correction were checked independently.
Numerical reference calculations use raw pairwise distances and direct
integration of the Student density raised to $q$. Rescaling before calculation
cancels in the entropy difference. Large-`df` calculations were checked against
the Gaussian/Shannon limit. Negative finite-sample values remain unclipped.
The formula is defined for $\nu>2$; stronger convergence moment conditions in
Theorem 6 are not asserted over the entire computational domain.

### Lequesne Rényi T1 and T2

The [criterion page](lequesne_renyi_student_criterion.md) follows equations
(7)–(10) of [Lequesne (2015)](https://doi.org/10.1063/1.4906014), checked directly
in the user-provided full text, pp. 489–493. T1 fixes $\nu$ and estimates variance;
choosing the unbiased variance makes its calculation equal to the univariate
Çadirci calculation. The paper allows a consistent variance estimator and does
not prescribe its denominator.

T2 uses $h_{\widehat q}(\widehat\nu,\widehat s^2)-\widehat H_{n,k,\widehat q}$,
with both shape and scale from the numerical MLE and
$\widehat q=(\widehat\nu-1)/(\widehat\nu+1)$. Tests separately check that the
fitted entropy order and MLE scale are used, rather than a fixed order or a
moment-matched scale. They also compare an actual fit with a separate SciPy fit.
Optimizer convergence is checked; that check is not a proof of a unique global
likelihood maximum. Fits with $\widehat\nu\le2$ are rejected, not clamped.
Theorem 2 uses the stronger condition $\nu>4$ (or $\widehat\nu>4$).

The lower-tail critical-region inequalities on p. 493 contradict the positive
alternative limit stated in Theorem 2. The implementation uses an upper-tail
rejection rule consistent with the entropy deficit. It does not embed Table 1,
claim agreement with its tabulated quantiles, or force finite-sample deficits to
be nonnegative. The scale notation in the density is used consistently:
standardization divides by $s$, not by $s^2$.

## Calibration, numerical boundaries and scope

All 21 criteria return a statistic and use `RightAlternative`; no automatic
p-value follows from executing a statistic. The fixed-null EDF discrepancies
are distribution-free under the continuous probability integral transform.
This does not extend to substituting estimated parameters into those classes.
Liao–Shimokawa here is a fixed-Student application of its EDF functional;
original fitted-Weibull/extreme-value critical values do not apply. Kuiper's
finite-sample multiplier is not itself a p-value algorithm.

EDF input can contain ties or constants and must be nonempty, finite and
one-dimensional. Student log-CDF and log-survival are computed separately;
AD, Liao–Shimokawa and Zhang consume logarithms directly. Underflow of a required
log probability or unrepresentable standardization raises `ValueError`.
A representable raw observation does not guarantee a representable tail.

Frequency input constraints also depend on the binning strategy. The default
CDF-space equal-probability rule includes an internal boundary in the right
cell and includes 1 in the last cell. Calibration preserves the selected bins
and count rule. The usual chi-square count limit is asymptotic, not an exact
finite-sample result; empty-bin infinities are intentional.

Composite criteria recompute estimates on each call and retain no fitted state.
Meintanis accepts ties but rejects constants; the entropy criteria require
distinct observations and $n>k$. Lequesne T2 needs at least four observations.
For fixed `df`, location and scale invariance permit standard Student Monte
Carlo samples, preserving `df`, `k` or the ECF weight and numerical settings.
For unknown `df`, Lequesne T2 requires an external shape-specific parametric
bootstrap and refitting on every replicate. The built-in sampler and storage
lookup reject that unspecified simulation shape; failed fits must not be
silently removed from calibration.

The report distinguishes source attribution from mathematical and executable
checks. The Meintanis author PDF, the Çadirci full preprint and the supplied
Lequesne PDF were read during implementation. References for generated kernels
are inherited from the shared EDF/frequency definitions and existing mathematical
reviews. This documentation audit does not claim fresh full-text verification
of every historical article: the DOI pages for Zhang, Liao–Shimokawa,
Cressie–Read and Lemeshko were inaccessible through the web tool. Formula and
numerical checks remain independent of that access limitation. None of these
checks establishes nominal test size or power against every alternative.

## Numerical and documentation verification

Verification on 2026-10-14 includes the following suites:

- `tests/distributions/test_student.py` and `test_student_correctness.py`:
  SciPy KS references, direct EDF integrals, Zhang equations, tails, sample
  validation, affine transformations, unchanged input and catalog completeness.
- `tests/statistics/test_student_composite.py`: closed-form ECF inversion,
  entropy density integrals and brute-force neighbours, sample/setting
  validation, invariance, calibration identity and large-`df` limits.
- `tests/statistics/test_lequesne_student.py`: both entropy formulas, actual
  and controlled MLE results, fitted order/scale, failed fits, explicit
  hypotheses, registration and bootstrap restrictions.
- `tests/statistics/test_edf.py` and `test_frequency.py`: independent shared
  formula references, high-precision Zhang values, SciPy power-divergence
  comparisons, zero counts and endpoint cases. These exercise common kernels;
  they are not additional Student-specific scientific derivations.

The documentation audit compares the runtime concrete-class catalog with
individual pages, requires scientific URLs and Python examples, checks every
catalog/navigation entry and local link, and executes the examples. It also
builds MkDocs to verify page rendering and math blocks. The measured results
and environment limitations are recorded below.

Results:

- **21/21 concrete criteria** have an individual page, a scientific reference,
  a Python example, an overview entry and a navigation entry.
- **22 Python examples executed successfully**, including the overview example;
  local documentation links resolved.
- **582 tests passed** across the six suites listed above.
- **MkDocs built successfully**; all 21 criterion pages contain generated
  `arithmatex` markup. This checks HTML generation, not browser-side MathJax
  rendering. Existing build warnings concern other documentation sections;
  none concern Student pages.

The tests and examples used an isolated `continuous` package namespace to avoid
executing its initializer. Ordinary package initialization currently fails
because `beta.py` imports `AbstractChi2Statistic` from `edf.py`, where that class
is no longer defined. Thus these results verify Student and shared kernels,
but do not certify that ordinary package imports work in the current checkout.
The documented examples use ordinary public imports and require that unrelated
initialization issue to be resolved when run without isolation.

To reproduce the isolated test run from the repository root:

```bash
.venv/bin/python - <<'PY'
import sys
import types
from pathlib import Path

name = "pysatl_criterion.statistics.goodness_of_fit.continuous"
package = types.ModuleType(name)
package.__path__ = [str(Path("src/pysatl_criterion/statistics/goodness_of_fit/continuous").resolve())]
sys.modules[name] = package

import pytest

raise SystemExit(pytest.main([
    "tests/distributions/test_student.py",
    "tests/distributions/test_student_correctness.py",
    "tests/statistics/test_student_composite.py",
    "tests/statistics/test_lequesne_student.py",
    "tests/statistics/test_edf.py",
    "tests/statistics/test_frequency.py",
    "-q",
    "--tb=short",
]))
PY
```
