# Mathematical verification of Beta statistics

Audit date: **2026-10-14**. Scope: all **23 concrete classes** exported by
`continuous/beta/beta_statistics.py`, including their shared EDF/frequency kernels and
`hypothesis_testing/beta_bootstrap.py`. See the [complete catalog](index.md)
for every class, short code and individual page.

## Conclusion and scope of the claims

The implemented definitions were checked algebraically and against independent
numerical references. The current catalog contains 18 generated criteria,
Neyman smooth and four fitted-Beta procedures: Ebner–Liebenberg, Raschke,
Riad–Abd El Mabood and Vasicek–Kullback–Leibler.
References identify published functionals or Beta-specific procedures; their
critical values cannot automatically be reused after fitting parameters.

Passing formula checks does not establish exact finite-sample significance,
uniform power, or numerical accuracy for every representable positive shape.

## Null hypotheses and estimation

$$
f_{a,b}(x)=\frac{x^{a-1}(1-x)^{b-1}}{B(a,b)},\qquad 0<x<1,\quad a,b>0.
$$

| Criteria                           | Hypothesis                                    | Estimation                                                                  |
|------------------------------------|-----------------------------------------------|-----------------------------------------------------------------------------|
| 13 EDF, 5 frequency, Neyman smooth | Both shapes fixed independently of the sample | None                                                                        |
| Ebner–Liebenberg                   | Both shapes unknown; support fixed            | Beta MLE on each call                                                       |
| Raschke                            | Both shapes unknown; support fixed            | Beta MLE, normal quantile transform, normal location/scale MLE on each call |
| Riad–Abd El Mabood; Vasicek–KL     | Both shapes unknown; support fixed            | Beta MLE; same weight/window in each bootstrap replicate                    |

Specified criteria use `Beta.DEFAULT.parse({"a": 2, "b": 5})`; fitted criteria
use `Beta.DEFAULT.parse({})`. No default or partially fixed shapes are accepted.
All critical regions are right-tailed. Samples must be finite, real and
one-dimensional; masked values are rejected rather than dropped.

## Generated formula checks

For sorted observations define $u_i=F_{a,b}(X_{(i)})$, $r_i=i-1/2$,
$D^+=\max_i(i/n-u_i)$, $D^-=\max_i(u_i-(i-1)/n)$ and
$W^2=1/(12n)+\sum_i(u_i-r_i/n)^2$. Counts are $O_j$ and $E_j=np_j>0$.

| Criterion                                                             | Definition                                                                                              | Source                                                                                                                                                                                                                                                                                    |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_beta_criterion.md)  | $D^+=\max_{1\le i\le n}(i/n-u_i)$                                                                       | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.1.                                                                                                                                                              |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_beta_criterion.md) | $D^-=\max_{1\le i\le n}(u_i-(i-1)/n)$                                                                   | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.1.                                                                                                                                                              |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_beta_criterion.md)            | $D=\max(D^+,D^-),\quad D^+=\max_i(i/n-u_i),\quad D^-=\max_i(u_i-(i-1)/n)$                               | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.1.                                                                                                                                                              |
| [Anderson–Darling](anderson_darling_beta_criterion.md)                | $A^2=-n-\frac1n\sum_{i=1}^n(2i-1)[\log u_i+\log(1-u_{n+1-i})]$                                          | [Anderson and Darling (1952), Asymptotic Theory of Certain Goodness of Fit Criteria Based on Stochastic Processes](https://doi.org/10.1214/aoms/1177729437); [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.2. |
| [Cramér–von Mises](cvm_beta_criterion.md)                             | $W^2=\frac{1}{12n}+\sum_{i=1}^n\left(u_i-\frac{i-1/2}{n}\right)^2$                                      | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §2.3.                                                                                                                                                              |
| [Liao–Shimokawa](liao_shimokawa_beta_criterion.md)                    | $L=\frac1{\sqrt n}\sum_{i=1}^n\frac{\max(i/n-u_i,\ u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}$                    | [Liao and Shimokawa (1999), A new goodness-of-fit test for Type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965).                                                                                              |
| [Watson](watson_beta_criterion.md)                                    | $U^2=W^2-n(\bar u-1/2)^2,\quad W^2=\frac1{12n}+\sum_i[u_i-(i-1/2)/n]^2$                                 | [Watson (1961), Goodness-of-fit tests on a circle](https://doi.org/10.1093/biomet/48.1-2.109).                                                                                                                                                                                            |
| [Kuiper](kuiper_beta_criterion.md)                                    | $V=D^++D^-,\quad D^+=\max_i(i/n-u_i),\quad D^-=\max_i(u_i-(i-1)/n)$                                     | [Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X), equations (1)–(3).                                                                                                                    |
| [Scaled Kuiper](kuiper_scaled_beta_criterion.md)                      | $V_s=\sqrt n(D^++D^-)$                                                                                  | [Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X), equations (1)–(3).                                                                                                                    |
| [Corrected Kuiper](kuiper_corrected_beta_criterion.md)                | $V_c=\left(\sqrt n+0.155+\frac{0.24}{\sqrt n}\right)(D^++D^-)$                                          | [Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X), equations (1)–(3).                                                                                                                    |
| [Zhang Z_K](zhang_k_beta_criterion.md)                                | $Z_K=\max_i\left[r_i\log\frac{r_i}{nu_i}+(n-r_i)\log\frac{n-r_i}{n(1-u_i)}\right],\quad r_i=i-\tfrac12$ | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.1).                                                                                                                                                    |
| [Zhang Z_A](zhang_a_beta_criterion.md)                                | $Z_A=-\sum_{i=1}^n\left[\frac{\log u_i}{n-i+1/2}+\frac{\log(1-u_i)}{i-1/2}\right]$                      | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.2).                                                                                                                                                    |
| [Zhang Z_C](zhang_c_beta_criterion.md)                                | $Z_C=\sum_{i=1}^n\left[\log\frac{1-u_i}{u_i}-\log\frac{n-i+1/4}{i-3/4}\right]^2$                        | [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.3).                                                                                                                                                    |
| [Pearson chi-squared](chi2_pearson_beta_criterion.md)                 | $X^2=\sum_j\frac{(O_j-E_j)^2}{E_j}$                                                                     | [Pearson (1900), On the criterion that a given system of deviations from the probable …](https://doi.org/10.1080/14786440009463897); [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x).                                    |
| [Freeman–Tukey](freeman_tukey_beta_criterion.md)                      | $FT=4\sum_j(\sqrt{O_j}-\sqrt{E_j})^2$                                                                   | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), power-divergence parameter $\lambda=-1/2$.                                                                                                                              |
| [Modified log-likelihood](modified_log_likelihood_beta_criterion.md)  | $MLL=2\sum_j E_j\log(E_j/O_j)$                                                                          | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=-1$.                                                                                                                                                           |
| [Neyman chi-squared](neyman_chi2_beta_criterion.md)                   | $N=\sum_j\frac{(O_j-E_j)^2}{O_j}$                                                                       | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=-2$.                                                                                                                                                           |
| [Cressie–Read](cressie_read_beta_criterion.md)                        | $CR=\frac95\sum_j O_j\left[(O_j/E_j)^{2/3}-1\right]$                                                    | [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=2/3$.                                                                                                                                                          |

The Beta EDF regression integrates the Beta density to obtain CDF values,
then checks all thirteen statistics at five shape pairs. KS additionally uses
SciPy's one-sided and two-sided tests; CvM, Watson and AD use piecewise numerical
EDF integrals. Zhang and Liao–Shimokawa use explicit rank formulas, independently
of production kernels. The shared-kernel suite adds high-precision Zhang
references and true endpoint limits. The five frequency implementations are
checked against SciPy power divergences with independently constructed bins.

With fixed shapes the continuous probability integral transform makes EDF null
laws shape-free. Ties can be evaluated, but rounding or discrete observations
change the continuous-null calibration. For fixed deterministic bins and positive
probabilities the usual frequency limit is $\chi^2_{J-1}$. The default
`EqualProbabilityBins()` uses a sample-size-dependent count; a fixed-degree
chi-square claim is not made for that rule. Empty observed cells give infinity
for modified log-likelihood and Neyman, which is mathematically intentional.
Calibration must preserve binning settings; storage keys serialize them.

## Specialized formula checks

### Neyman smooth

For fixed positive integer $k$ and $u_i=F_{a,b}(X_i)$,

$$T_k=\frac1n\sum_{j=1}^k\left(\sum_i\sqrt{2j+1}P_j(2u_i-1)\right)^2.$$

Shifted normalized Legendre polynomials have zero means and identity covariance
under a uniform null, yielding a $\chi^2_k$ limit for fixed $k$. The recurrence
is checked against explicit polynomials of orders 1–4 and endpoint identities
at higher order. The result is shape-free under a fully specified Beta null,
but fixed $k$ does not give omnibus consistency. This applies
[Neyman (1937)](https://doi.org/10.1080/03461238.1937.10404821) through the PIT.
`k != 4` is rejected by the storage-calibration guard; use matching Monte Carlo
simulation. The class does not fit unknown shapes.

### Ebner–Liebenberg

With both shapes estimated by MLE and $c_j=(\hat a+\hat b)X_j-\hat a$,
the statistic is $n\int_0^1[n^{-1}\sum_j c_j1\{X_j\ge t\}
-t^{\hat a}(1-t)^{\hat b}/B(\hat a,\hat b)]^2dt$.
The sorted cumulative implementation matches equation (3) of the
[full preprint](https://arxiv.org/pdf/2009.13995), independently checked by
quadrature and a quadratic pairwise sum. Beta-function ratios preserve the
cross and constant terms. Require at least two distinct observations strictly
inside the support. Refitting occurs on each bootstrap replicate; no fitted
state is retained. The paper's bootstrap and consistency statements have
regularity conditions, not an exact finite-sample distribution-free law.

### Raschke

The author manuscript's sections 4–5 prescribe Beta MLE, normal quantile
transformation, normal MLE and corrected AD. The implementation uses normal
variance divisor $n$ and correction $1+0.75/n+2.25/n^2$. A separate weighted
EDF integral checks the resulting value. At least three nonconstant interior
observations are required; with two, normal standardization makes the scores
$(-1,1)$ and the statistic constant. Separate Beta tails avoid `1-CDF`
cancellation; unrepresentable normal transforms raise an error.
The paper's normal critical values are approximate. The supplied Beta bootstrap
repeats both fits and is a different calibration; its validity is not implied
by the Ebner–Liebenberg theorem. See [Raschke's page](raschke_beta_criterion.md)
and [author manuscript](https://www.researchgate.net/publication/220504959_The_Biased_Transformation_and_Its_Application_in_Goodness-of-Fit_Tests_for_the_Beta_and_Gamma_Distribution).

## Numerical boundaries

EDF underflow is handled with the
[DLMF 8.17.8 identity](https://dlmf.nist.gov/8.17.E8); a failed hypergeometric
evaluation raises an error. True endpoints can legitimately yield infinity.
The fitted criteria require interior, nonconstant samples and propagate fitting
failures. Calibration must preserve the hypothesis and any binning/order settings.

## Package layout and import compatibility

Beta follows the uniform package layout: `beta/beta_statistics.py` contains
implementations, `beta/__init__.py` explicitly re-exports the public classes,
and `beta/__init__.pyi` declares their types. The former `beta.py` and
`beta.pyi` are replaced by these package files.
All 25 public classes (23 concrete and two abstract) retain the module path
`pysatl_criterion.statistics.goodness_of_fit.continuous.beta`. Regression checks
cover identity between the package and implementation, old pickle references,
round trips and unique criterion discovery. Private catalogs live in the
implementation module and are not re-exported from the package.

## Type declarations

`src/pysatl_criterion/statistics/goodness_of_fit/continuous/beta/__init__.pyi` declares
all 23 concrete criteria and both abstract bases, including generated EDF and
frequency classes. Constructor options `weight`, `window`, `k` and `binning`,
scalar results, NumPy array arguments and class-level numerical helpers are typed.
`tests/distributions/test_beta_stub.py` checks exports, inheritance, method
parameter names and keyword-only defaults against runtime definitions.
The generated stub is maintained with `python scripts/generate_beta_stub.py`;
`python scripts/generate_beta_stub.py --check` verifies that it is current
without importing the package. Generation helpers are methods of
`BetaStubGenerator`, and type annotations are preserved through formatting.
A focused mypy check of the stub and usage examples passed with
`--follow-imports=skip` to isolate unrelated package typing issues.

## Numba execution and optimization

All 23 criteria execute numerical reductions through class methods decorated
with `@jit(nopython=True, cache=True)`, without `fastmath`. This does not mean
that SciPy distribution transforms or maximum-likelihood fitting are JIT compiled.

| Criteria | Compiled computation | Work retained in Python/SciPy |
|---|---|---|
| 13 EDF | Shared EDF and log-tail reduction kernels | Validation, sorting and Beta log tails |
| 5 frequency | Shared frequency divergence kernels | Binning and expected probabilities |
| Ebner–Liebenberg | Tail sums and statistic reduction | MLE, sorting and incomplete Beta functions |
| Neyman smooth | Legendre recurrence and squared-component sums | Validation and Beta CDF |
| Riad–Abd El Mabood | Residual coefficients, compensated pair sum and fallback integrand | MLE and adaptive quadrature controller |
| Vasicek–KL | Fused entropy/log-density reduction | MLE, sorting and scalar `betaln` |
| Raschke | Two-pass normal standardization and AD reduction | Beta MLE and Beta/normal distribution functions |

KL now computes window gaps, log spacings and log densities in a single loop,
without materializing those arrays. The Laplace fallback computes the minimum
and shifted sample once, and calls a compiled integrand through `quad(args=...)`.
Raschke standardizes in a compiled two-pass calculation using the MLE variance
with denominator n. All new helpers are class methods; the EDF log-probability
callback was also moved into `AbstractSpecifiedBetaGofStatistic`.

`tests/distributions/test_beta_numba.py` checks that every concrete criterion
executes a nopython kernel, compares the new reductions with independent SciPy
references, and covers degenerate transformations and input preservation.
The existing integral, tail-underflow, negative-KL and quadrature-fallback
regressions remain in the complete run below.

Local timing on 2026-10-14: median of seven batches of ten complete
`execute_statistic` calls, after JIT warm-up, on Beta(2,5) samples generated
with `np.random.default_rng(234)` separately for each n. Times include fitting.

| Criterion | n | Before, microseconds | After, microseconds |
|---|---:|---:|---:|
| Vasicek–KL | 100 | 137.2 | 58.4 |
| Vasicek–KL | 1000 | 214.3 | 94.0 |
| Raschke | 100 | 288.1 | 264.1 |
| Raschke | 1000 | 913.7 | 832.3 |
| Riad–Abd El Mabood | 100 | 71.3 | 68.1 |
| Riad–Abd El Mabood | 1000 | 2013.8 | 1933.6 |

These are local measurements, not portable performance guarantees. Small
Laplace differences may reflect timing variability; its dominant pairwise
kernel was already compiled. First-call compilation cost is excluded.
Observed KL differences were below 6e-15; the other recorded values agreed
at float64 precision for these benchmark samples.

## Added published criteria: independent checks

### Riad–Abd El Mabood (2018)

See the [Laplace criterion](riad_abd_el_mabood_beta_criterion.md) for the complete
formula and [original DOI](https://doi.org/10.17654/AS053020165).
The implementation uses MLE shapes in the Kummer residual with weight $e^{-\rho t}$.
Integrating each pair uses $\int_0^\infty t^k e^{-st}dt=k!/s^{k+1}$.
Tests compare the resulting sum with independent numerical integration of
$L_n,L_n',L_n''$, for multiple weights, shapes and sample sizes.

**Source discrepancy:** the expanded RF expression in Section 4 of
[Ebner–Liebenberg (published text)](https://publikationen.bibliothek.kit.edu/1000134155/118245046)
prints $\hat\alpha^2$ in the cross-term denominator. From
$A_i=x_i(x_i-1)/\hat\alpha$ and $B_i=1-x_i/\hat\mu$, the cross term
$A_iB_j+A_jB_i$ contains one factor $1/\hat\alpha$, not two.
We implement the squared differential-equation residual, independently
verified by quadrature; the printed expansion is not used verbatim.
Here $\hat\mu=\hat\alpha/(\hat\alpha+\hat\beta)$, as defined in that
published comparison. It is generally different from the sample mean.
The original 2018 full text was unavailable; exact replication of its tables
is not claimed. Default weight 2 is documented and included in configuration.

### Alizadeh Noughabi–Arghami (2013)

See the [KL criterion](kullback_leibler_beta_criterion.md) and
[original DOI](https://doi.org/10.1080/00949655.2012.667100).
The original full text was unavailable. The general Vasicek/MLE formula was
verified in [Section 2 of the same author's later paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041611/).
Substituting the Beta log density yields the documented statistic. This is
not the previously removed difference between theoretical and sample entropy.
Tests use SciPy's explicit Vasicek estimator and independent Beta log-density
calculation, including boundary window indices, reflection and negative values.
The library window default is explicit and is not attributed to the 2013
paper. Zero window spacings raise an error; finite negative estimates remain
unchanged. Bootstrap preserves settings and refits both shapes in every draw.

Both new criteria reject large values. Their tests cover configuration,
invalid settings/samples, no input mutation, permutation invariance and
reproduction of bootstrap p-values. This validates definitions and computation,
not published power tables or exact finite-sample significance.

## Verification and reproducibility

The audit runs all Beta suites plus the shared EDF/frequency suites:
`tests/distributions/test_beta*.py`, `tests/statistics/test_beta_dynamic.py`,
`test_beta_mathematical_validation.py`, `test_beta_published.py`,
`test_edf.py`, and `test_frequency.py`.
These cover formula references, sample constraints, hypotheses, endpoints,
small and large shapes, unchanged inputs, fitting and bootstrap reproduction.

The documentation audit compares runtime concrete classes with all 23 pages,
checks scientific URLs, Python examples, overview/navigation coverage and local
links, then executes every page example. MkDocs checks HTML generation and
math markup; it does not prove browser-side MathJax rendering.

Historical source access is not uniform: the Ebner–Liebenberg full preprint
and Raschke author manuscript were accessible. DOI access to Zhang,
Liao–Shimokawa, Cressie–Read, Lemeshko and Neyman failed through the
web tool. Their links and shared definitions are documented, but this report
does not claim fresh full-text verification of those articles.

Normal package initialization currently fails on stale imports of
`AbstractChi2Statistic` and `AbstractCrammerVonMisesStatistic` from `edf.py`
in the common initializer and unrelated distribution modules. The tests and
examples therefore use an isolated `continuous` namespace, without introducing
compatibility aliases into production. This limitation applies to the public
import path and is not concealed by the numerical results.

Reproduce the isolated test run from the repository root:

```bash
.venv/bin/python - <<'PYCODE'
import sys
import types
from pathlib import Path
import pytest
name = "pysatl_criterion.statistics.goodness_of_fit.continuous"
package = types.ModuleType(name)
package.__path__ = [str(Path("src/pysatl_criterion/statistics/goodness_of_fit/continuous").resolve())]
sys.modules[name] = package
from pysatl_criterion.statistics.goodness_of_fit.continuous import beta
for export in beta.__all__:
    setattr(package, export, getattr(beta, export))
raise SystemExit(pytest.main([
    *map(str, sorted(Path("tests/distributions").glob("test_beta*.py"))),
    "tests/statistics/test_beta_dynamic.py",
    "tests/statistics/test_beta_mathematical_validation.py",
    "tests/statistics/test_beta_documentation.py",
    "tests/statistics/test_beta_published.py",
    "tests/statistics/test_edf.py",
    "tests/statistics/test_frequency.py", "-q", "--tb=short",
]))
PYCODE
```

Measured results for the extended catalog:

- **680 tests passed** in the isolated run described above, including the independent
  formula suites and the Numba execution regressions.
- **23/23 criteria** have a dedicated page, scientific reference, Python example,
  overview entry and navigation entry.
- **25 Python example blocks executed successfully**, each in a fresh namespace.
- All Beta Markdown local links resolve.
- MkDocs built successfully. Existing warnings concern other documentation sections.
- Ruff passed for the implementation and both new numerical test suites.

Ordinary package imports remain blocked by the unrelated obsolete imports
listed above. No compatibility aliases were added.
