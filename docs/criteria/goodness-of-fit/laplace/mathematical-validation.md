# Laplace: mathematical validation

## Scope and sources

The [catalog](index.md) contains 38 concrete criteria, each with a separate page:
13 generated EDF criteria, five generated frequency criteria, Greenwood and
19 fitted specialist criteria. The original 19 use a fully specified Laplace(t, s) null.
The added 19 use a composite location-scale null; see the fitted-test audit below.
The checks below distinguish independent numerical validation from bibliographic
attribution. No Laplace-specific power or optimality claim is made.

The definitions of D, W², A², V and U² follow Section 2 and Case 0 of
[Stephens (1974), EDF Statistics for Goodness of Fit and Some Comparisons](https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf).
Greenwood uses the n+1 spacings, including both endpoints, in equations (1)–(2) of
[Does, Helmers and Klaassen (1988)](https://ir.cwi.nl/pub/1694/1694D.pdf).
These sources concern continuous, fully specified null distributions.
No Laplace-specific power or optimality claim is made.

## Formulas and independent checks

Let z=(x-t)/s, F(x)=exp(z)/2 for z<0 and F(x)=1-exp(-z)/2 otherwise.
Let u_i=F(x_(i)), i=1,…,n, with u_0=0, u_(n+1)=1.
The original six statistics use the unmodified, unscaled definitions below.
Scaled and corrected Kuiper variants are listed separately.

| Code | Formula | Independent verification |
|---|---|---|
| KS | D=max(D+,D−), D+=max(i/n−u_i), D−=max(u_i−(i−1)/n) | SciPy CDF and direct maxima; all three directions |
| CVM | W²=1/(12n)+Σ(u_i−(2i−1)/(2n))² | SciPy probabilities and numerical integration of n(F_n(u)−u)² |
| AD | A²=−n−Σ((2i−1)/n)(log u_i+log(1−u_(n+1−i))) | SciPy log probabilities, weighted numerical integration, analytic extreme-tail limits |
| KUI | V=D++D− | Direct maxima using SciPy probabilities |
| WAT | U²=W²−n(mean(u)−1/2)² | Direct formula and integration of centered EDF discrepancy |
| GRW | G=Σ(u_i−u_(i−1))², i=1,…,n+1 | Direct spacings including endpoints and exact tied-sample example G=1/4 |

### Additional EDF criteria

Use the same sorted probabilities and define r_i=i−1/2. Each short code below
has the suffix `_LAPLACE_GOODNESS_OF_FIT` in the registry.

| Code and page | Formula | Independent verification |
|---|---|---|
| [KS_PLUS](kolmogorov_smirnov_d_plus_laplace_criterion.md) | D+=max(i/n−u_i) | Direct maximum of SciPy Laplace probabilities |
| [KS_MINUS](kolmogorov_smirnov_d_minus_laplace_criterion.md) | D−=max(u_i−(i−1)/n) | Direct maximum of SciPy Laplace probabilities |
| [KUIPER_SCALED](kuiper_scaled_laplace_criterion.md) | √n(D++D−) | Direct scaling of independent maxima |
| [KUIPER_CORRECTED](kuiper_corrected_laplace_criterion.md) | (√n+0.155+0.24/√n)(D++D−) | Direct scaling of independent maxima |
| [LIAO_SHIMOKAWA](liao_shimokawa_laplace_criterion.md) | Σ max(i/n−u_i, u_i−(i−1)/n)/√(u_i(1−u_i))/√n | Direct weighted formula using SciPy probabilities |
| [ZHANG_ZK](zhang_k_laplace_criterion.md) | max[r_i log(r_i/(nu_i))+(n−r_i)log((n−r_i)/(n(1−u_i)))] | Direct likelihood-ratio expression |
| [ZHANG_ZA](zhang_a_laplace_criterion.md) | −Σ[log(u_i)/(n−r_i)+log(1−u_i)/r_i] | Direct logarithmic expression |
| [ZHANG_ZC](zhang_c_laplace_criterion.md) | Σ[log((1−u_i)/u_i)−log((n−i+1/4)/(i−3/4))]² | Direct log-odds expression |

The added pages retain the shared algorithms' scientific references: Smirnov
for directional KS, Stephens/Lemeshko for Kuiper scaling, Zhang (2002),
equations (3.1)–(3.3), for Z_K/Z_A/Z_C, and Liao–Shimokawa (1999).
The latter is an adaptation to the fixed Laplace null; fitted-Weibull tables
from that paper do not apply. The additional references are inherited from the
shared algorithms' documentation; their full texts were not independently
reverified during this documentation update (the DOI retrievals were unavailable).
The direct-formula tests verify implementation, not published power claims.

### Frequency criteria

Let O_j be observed counts and E_j=np_j>0 expected counts for exhaustive bins.
With equal-probability bins E_j=n/m. Explicit or adaptive strategies can have
unequal expected counts. For each of the five classes, independent counts
(2,3,4,5) and equal expected counts are compared with SciPy `power_divergence`.

| Code and page | Formula | SciPy λ | Empty observed bin |
|---|---|---:|---|
| [CHI2_PEARSON](chi2_pearson_laplace_criterion.md) | Σ(O_j−E_j)²/E_j | 1 | contributes E_j |
| [FREEMAN_TUKEY](freeman_tukey_laplace_criterion.md) | 4Σ(√O_j−√E_j)² | −1/2 | contributes 4E_j |
| [MODIFIED_LOG_LIKELIHOOD](modified_log_likelihood_laplace_criterion.md) | 2Σ E_j log(E_j/O_j) | −1 | +infinity |
| [CHI2_NEYMAN](neyman_chi2_laplace_criterion.md) | Σ(O_j−E_j)²/O_j | −2 | +infinity |
| [CRESSIE_READ](cressie_read_laplace_criterion.md) | (9/5)Σ O_j[(O_j/E_j)^(2/3)−1] | 2/3 | zero contribution in this expression |

These are the shared power-divergence definitions documented with the Pearson
and Cressie–Read references on the criterion pages. Cressie–Read fixes λ=2/3;
there is no `lambda_` constructor option. Modified log likelihood uses E_j in
the weight and differs from the ordinary G statistic (λ=0).

### Extreme-value checks for the original six criteria

For (-a,0,a), A² approaches 2a/3+8log(2)/3−3 as a grows.
At a=10^308 the result remains representable. This detects premature overflow
when tail logs are added before normalization. Watson at n=1 equals 1/12
for every observation; equally spaced midpoint probabilities give 1/(12n).
Location-scale transformations are checked independently for all six criteria.

## Changes and compatibility

- Converted `continuous/laplace.py` into a package with explicit exports and a
  generated `__init__.pyi`; generator supports `--check` without importing the package.
- Preserved all six public class names, short codes and the `continuous.laplace`
  module identity used by pickle. The abstract family no longer declares hypotheses;
  concrete classes and factory specifications declare both parameters fixed.
- All 13 EDF and five frequency criteria use shared factories and kernels.
  Greenwood is the only manually declared concrete class. Generated KS retains
  its existing constructor through the factory compatibility option; its former
  call to the removed multi-argument KS interface is corrected.
- `do_execute_statistic(sample)` remains an observation-based compatibility entry
  point. Private `_calculate_statistic` helpers were removed; benchmark scripts now
  use the public `execute_statistic` API. `mode` remains a compatibility setting.
- Analytic CDF and log probabilities and Greenwood spacings are static cached
  Numba methods. No numerical helpers remain outside classes.
- Shared AD log evaluation now weights each tail before adding, avoiding intermediate
  overflow. The mathematical definition and calibration scale are unchanged.
- A factory validation callback retains Laplace's rejection of strings and other
  nonnumeric inputs. Masked observations are now rejected instead of silently used.

## Input and calibration limits

Nonempty one-dimensional finite real samples are required. Singletons, ties,
constant samples and gaps of any width are allowed. The support is the entire
real line. EDF and Greenwood require no binning; ties give zero spacings.
Frequency criteria use a configurable binning strategy and positive expected
counts. Empty observed bins are handled as specified in the table above. The exact continuous null calibration assumes continuous observations;
rounding-induced ties require a matching observation model in simulation.

For EDF and Greenwood, the PIT gives a parameter-free null law at fixed n when
both parameters are known. Frequency calibration must also match `binning`,
including adaptive bin selection; arbitrary binning is not parameter-free. The
default has no minimum expected-count constraint, so a chi-squared approximation
is not automatically justified. See [binning calibration](../binning/calibration.md).
Monte Carlo must call the same statistic, including the KS direction. Stored KS
calibration is rejected for LEFT/RIGHT because its key does not encode direction.
Fitting t or s outside this API does not justify using the known-parameter null law;
a fitted test would need a separately supported hypothesis and refitting in every
simulation. All 19 criteria use the right rejection tail. Greenwood's right-tail
form detects clustering; universal consistency is not asserted.

CDF rounding can erase tiny tail spacings. AD, Liao–Shimokawa and Zhang K/A/C
use analytic log tails instead of reconstructing rounded tail probabilities. Standardization avoids overflow in x−t when the quotient
is representable; a result beyond float64 range is still +infinity. No clipping
or Python high-precision fallback is used. Warm timing includes validation,
sorting, CDF evaluation and the shared kernel.

## Reproducible checks

```text
.venv/bin/python -m pytest tests/distributions/test_laplace.py tests/distributions/test_laplace_correctness.py tests/statistics/test_laplace_architecture.py tests/statistics/test_edf.py -q
.venv/bin/python scripts/generate_laplace_stub.py --check
.venv/bin/python -m mkdocs build --strict
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_laplace_pipeline.py
```

The ordinary pytest run is blocked during collection by an existing unrelated
import of `AbstractChi2Statistic` from `edf` in `exponentiated_weibull.py`.
An isolated run bypasses only `continuous/__init__.py` using a package placeholder;
it loads real Laplace, factory, EDF, distribution and resolver implementations.
Registry assertions in that run verify the Laplace subtree, not successful loading
of the entire project's registry. This distinction also applies to benchmarks.

### Results (2026-10-16)

- **478 tests passed in an isolated run** covering Laplace, shared EDF kernels
  and Uniform/Pareto factory regressions. Checks include independent formulas,
  integrals, SciPy power divergences, extreme tails, input validation, KS options,
  pickle, registry uniqueness, nopython dispatch, stub and catalog coverage.
- Ruff on changed Python files and generated stub: passed. Stub `--check`: passed.
- MkDocs normal build: passed. Strict build: blocked by 85 pre-existing warnings
  elsewhere (including missing normality and Weibull audit links); no Laplace warning.
- Instance pickle now serializes plain parameter values and reconstructs the
  supported Laplace schema, avoiding the shared immutable mapping's pickle error.

The following measurements cover the original six criteria only. No timing claim
is made for the 13 additional variants. Measured on the local Python 3.11 environment, n=10,000, seed 42, best of three
batches of 200 calls. Both paths include sorting; the reference uses SciPy CDFs
and independent NumPy formulas. These are local measurements, not guarantees.

| Code | First call, ms | Warm public call, µs | SciPy reference, µs | Speed ratio |
|---|---:|---:|---:|---:|
| AD | 83.422 | 370.70 | 498.09 | 1.34× |
| CVM | 4.595 | 301.00 | 415.27 | 1.38× |
| GRW | 1.955 | 294.14 | 414.08 | 1.41× |
| KS | 3.229 | 320.25 | 424.47 | 1.33× |
| KUI | 3.636 | 319.63 | 423.42 | 1.32× |
| WAT | 3.032 | 317.74 | 418.28 | 1.32× |

First-call time includes Numba initialization/cache loading and is excluded from
warm measurements; it is not a measurement of compiler time alone. All numerical
kernels use `nopython=True, cache=True`. Avoiding duplicate validation and sorting
in the factory callback improved the full generated-criterion path.


## Documentation coverage

All 19 concrete classes have individual pages linked from the catalog and MkDocs
navigation. Each page states the fixed null, formula, input limits, rejection
direction, calibration requirements, an executable example and scientific sources.
`mathematical-validation.md` summarizes verification; it does not replace those pages.

## Fitted-test audit

The added formulas follow the original article citations in each class and
[Desgagné–Lafaye de Micheaux–Ouimet, comparison, Section 3](https://arxiv.org/abs/2007.06154).
[Exact definitions](fitted-definitions.md) record all estimator and normalization choices.

Independent checks in `tests/statistics/test_laplace_published.py` cover:

- Meintanis: numerical integration of the squared empirical-characteristic-function discrepancy, both weights and both parameter estimators.
- Choi–Kim: direct spacings, van Es harmonic correction and independent regression slopes for Correa.
- AP: entropy spacings and numerical integration of the absolute deviation of the reconstructed piecewise-linear quantile; corrected and MLE variants.
- Gel, GV and rescaled V3/V4: direct sample central moments and the correct centering of absolute deviations.
- A_rat: direct products over every admissible integer window; optional log transform checked separately.
- AJ: independent SciPy fitted CDF differences.
- DLO: even/odd corrections checked at n=4, 5, 20, 21, including the zero-log-zero convention.
- Data-driven smooth: independent population integrals in observation coordinates, covariance solve and Schwarz selection.
- All added classes: hypothesis capabilities, registry/pickle, affine invariance, permutation invariance, finite-real input validation, no sample mutation and calibration sampler resolution.

### Explicit adaptations and limits

Entropy windows default to m=1 rather than silently extrapolating power-optimal tables.
AP_y rejects ties; other spacing tests use their limiting zero/infinite statistic when
a required spacing vanishes. A_rat uses the additional n/2 upper window bound from
the comparison, and offers an explicit log-scale option. DLO starts at n=4 because
its finite-sample odd-n variance correction is invalid at n=3.

The smooth moment components are rescaled to asymptotic unit variance (63 and 1188).
Their reconstruction uses the moment components and variances reported in
[Batsidis, Economou and Bar-Lev (2022), Section 3.3](https://www.ajs.or.at/index.php/ajs/article/download/1251/754),
and is independently checked using Laplace moments. The original Best et al. full
text was unavailable; no unverified DD formula is attributed to it.
The data-driven class is explicitly a Laplace specialization of the general
[Janic–Ledwina score construction](https://www.impan.pl/~ledwina/Janic-Ledwina-24-09-08.pdf),
not a replication of the DD comparator in Best et al.

Most arithmetic runs in cached nopython kernels. Meintanis has a quadrature fallback
for cancellation/extreme weights. Data-driven population matrices are computed once
per instance with SciPy, outside the sample kernel. Fitting/validation remains NumPy.
