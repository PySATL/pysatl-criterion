# Mathematical verification of Gamma statistics

Audit date: 2026-10-17. The scope is **all 28 concrete Gamma classes** exported
by `gamma/gamma_statistics.py`: 13 generated EDF, five generated frequency and 10 local classes.
The [catalog](index.md) links every criterion to a page with its exact formula,
constructor example and scientific source. Abstract infrastructure is excluded.

## Null hypothesis and conventions

Shape `alfa` and rate `beta` are fixed, finite and positive; origin is zero.
SciPy's scale is `1/beta`. No parameter estimation is performed.
Let $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$.
For spacing statistics include $u_0=0$ and $u_{n+1}=1$.
The probability integral transform makes these $u_i$ uniform order statistics
under the specified continuous null. This argument supports applying general
uniform EDF/spacing formulas to Gamma; it does not transfer fitted-model tables.

## Formula checks

| Criteria (number of public classes) | Checked definition | Source and qualification |
|---|---|---|
| Kolmogorov–Smirnov D+, D−, D (3) | Positive/negative supremum discrepancies; D is their maximum, without scaling | [Zhang (2002), §2.1](https://doi.org/10.1111/1467-9868.00337) |
| Anderson–Darling (1) | Weighted logarithmic sum, factor 1/n and subtraction of n | [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437); [Zhang, §2.2](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf) |
| Cramér–von Mises (1) | Squared deviations from half-ranks plus 1/(12n) | [Zhang, §2.3](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf) |
| Watson (1) | Centered Cramér–von Mises, subtracting $n(\bar u-1/2)^2$ | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109); [Lemeshko et al. (2014), eq. (6)](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf) |
| Kuiper, scaled, corrected (3) | $V=D^++D^-$; $\sqrt nV$; $(\sqrt n+0.155+0.24/\sqrt n)V$ | [Lemeshko et al., eqs. (1)–(3)](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf) |
| Zhang K, A, C (3) | Half-rank likelihood ratios and quarter-rank log odds | [Zhang, eqs. (3.1)–(3.3)](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf); Z_C implements the paper's stated approximate form |
| Liao–Shimokawa | Sum of local EDF maxima weighted by $[u_i(1-u_i)]^{-1/2}/\sqrt n$ | [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965); formula also explicit in [Al-Zahrani (2012), eq. (13)](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf) |
| Pearson, Freeman–Tukey, modified log-likelihood, Neyman, Cressie–Read (5) | Power divergences at $\lambda=1,-1/2,-1,-2,2/3$ | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x); explicit zero-count limits |
| Likelihood ratio (1) | $2\sum_{O_j>0}O_j\log(O_j/E_j)$, power-divergence limit at zero | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| Greenwood (1) | Sum of squares of all n+1 spacings | [Does, Helmers and Klaassen (1988), eqs. (1)–(2)](https://ir.cwi.nl/pub/1694/1694D.pdf) |
| Moran (1) | Negative sum of log spacings scaled by n | [Moran (1951)](https://doi.org/10.1111/j.2517-6161.1951.tb00079.x), bibliographic background; the retained normalization is qualified below |
| Probability plot correlation (1) | One minus correlation with Gamma quantiles at $(i-3/8)/(n+1/4)$ | [Filliben (1975)](https://doi.org/10.1080/00401706.1975.10489279), methodological background only; Gamma/Blom variant is a library adaptation |
| Graph edge count, maximum degree, average degree, components, clique number, independence number (6) | Exact graph with edges iff $|u_i-u_j|<(\max u-\min u)/10$ | Related methodology: [Ebner, Nestmann and Schulte (2020 preprint)](https://arxiv.org/abs/1812.08638); this range-based construction and two-tail convention are library definitions |

## Findings and qualifications

No mathematical formula change was required for the definitions checked.
This is a verification of statistic calculation, not certification of type-I
error, consistency, or power for all alternatives.

- **Liao–Shimokawa uses the shared EDF kernel.** The original
  Liao–Shimokawa paper studies fitted Weibull/extreme-value models. Its critical
  values cannot be reused for the fixed Gamma hypothesis.
- **Moran has a normalization offset.** The code returns
  $-\sum\log(nD_j)$, exceeding $-\sum\log((n+1)D_j)$ by
  $(n+1)\log((n+1)/n)$. The identity was verified algebraically and on equal
  spacings. This does not change rankings at fixed n, but changes critical values.
  The primary paper's exact normalization was not verified from accessible full
  text; the citation is not used to claim that it specifies the library's factor n.
- **PPCC is not an omnibus fixed-parameter Gamma test.** Correlation eliminates
  positive scale and translation of observations. Such alternatives can therefore
  leave the statistic unchanged. At n=2 it is zero in exact arithmetic. Its null
  distribution depends on shape; normality PPCC tables are not applicable.
- **The six graph statistics are local adaptations.** Graph quantities were
  verified against independent enumeration. The referenced paper does not validate
  this exact threshold. Affine transformations of u scale both distances and h
  identically; compressed uniform positions inside [0,1] give a concrete blind
  spot. This is an inference from the implemented rule. Average degree is exactly
  twice edge count divided by n, so the two contain the same information at fixed n.
- **Tail selection is part of the definition.** All nongraph statistics reject
  large values; graph statistics use both tails. This includes D− and D+ despite
  their CDF-direction names. Greenwood's chosen right tail does not test excessive
  regularity of spacings.

Bibliographic verification is distinguished from full-text formula verification.
Zhang, Lemeshko et al., and Does et al. were checked in accessible papers linked
above. For Liao–Shimokawa, the explicit expression was checked in Al-Zahrani's
published application, rather than claiming access to the original full text.
PPCC and graph citations provide background, not evidence for the exact local
Gamma constructions. Every criterion page states its applicable qualification.

## API and documentation corrections

Previously there were 18 criterion pages for 29 public classes. Eleven pages,
a complete index and this report have been added, and all pages are in MkDocs
navigation. All 28 examples are executed in the test suite. Broken links to the
absent `gamma-statistics-audit.md` have been replaced.

Generated frequency constructors use `binning=EqualProbabilityBins(count=k)`,
not `bins=k`. Their default is `PowerCount()` (ceil(2*n**0.4), bounded to 2…1000),
not eight bins. Cressie–Read power is fixed at 2/3. Only the local likelihood-ratio
class retains `bins=k` and the default eight bins. Generated binning configurations
are included in storage keys, whereas likelihood-ratio storage remains blocked.
Watson and Kuiper use the generated short codes `WATSON` and `KUIPER`.
Old tests and documentation expecting the pre-generation interface were corrected.

The companion `gamma/__init__.pyi` declares all 31 public classes (28 concrete and three
abstract), including dynamically generated criteria, frequency `binning`, legacy
KS options and the local likelihood-ratio `bins` constructor. Regenerate it with
`.venv/bin/python scripts/generate_gamma_stub.py`; use `--check` to detect stale
declarations without importing the package. Public class coverage and declared
constructor signatures were checked against the runtime implementation.

The package follows the same layout as `continuous/uniform`: numerical
implementations live in `gamma/gamma_statistics.py`, explicit public re-exports
in `gamma/__init__.py`, and generated type declarations in `gamma/__init__.pyi`.
Imports from `continuous.gamma` and class module identities remain unchanged,
including pickle lookup paths for classes. Instance serialization is outside this
check: parameter values currently contain an unpicklable mapping proxy. Internal catalogs are in `gamma_statistics`.

## Numerical verification

The isolated focused run passed **960 tests**. Ruff passed for the changed Python
files. MkDocs built successfully; existing missing-link warnings belong to other
distribution sections. All local Gamma links resolve. The focused suite checks:

- Independent closed-form EDF definitions at four shape/rate pairs, with a
  prescribed probability sample, for all 13 generated classes.
- Numerical integration of the step EDF for Anderson–Darling, Cramér–von Mises
  and Watson; SciPy one-sample KS in each direction.
- SciPy power-divergence comparisons with positive counts and analytic limits
  with empty cells for all six frequency discrepancies, including likelihood ratio.
- Both endpoint spacings, Moran's offset, ties, zeros, and exact exponential
  survival identities for the shape-one Gamma upper tail.
- Independent Pearson correlation for PPCC; its affine blind spot and n=2 case.
- NetworkX/brute-force clique and independent-set references for all graph
  outputs, including ties, singleton/constant samples and ten random samples.
- Rate/unit invariance for every class, invalid inputs and parameters, unchanged
  samples/state, class generation, hypothesis declarations and pickle identity.
- Page/catalog/navigation coverage, all 28 Markdown examples and module doctests.

Logarithmic implementations avoid cancellation but remain limited by SciPy/float64
Gamma probability evaluation. Extreme log probabilities can become infinite;
CDF saturation can merge graph positions. No claim of arbitrary-precision tail
accuracy is made. Fixed-bin chi-square approximations need adequate expected
counts; adaptive bins require calibration of their selection process.

### Numba optimization

All 28 concrete criteria now dispatch to numerical kernels compiled with
`@jit(nopython=True, cache=True)`. The 18 generated EDF/frequency classes already
used compiled shared kernels. The 10 local classes now have static compiled
`do_execute_statistic` methods. Validation, Gamma CDF/log-CDF/log-survival and
quantile calls remain outside JIT because they use SciPy. There are no module-level
functions in `gamma/gamma_statistics.py`; validation and sorting helpers belong to the base class.
`gamma/__init__.pyi` was regenerated, and its generator removes JIT decorators from stubs.

Greenwood and Moran accumulate results without temporary spacing,
rank or weight arrays. PPCC uses scaled two-pass moments without modifying its
arguments. The likelihood ratio handles zero counts explicitly. Graph kernels
use the ordering of CDF positions: sliding windows count neighbors and cliques,
adjacent gaps determine components, and greedy selection determines the maximum
independent set. Each graph kernel takes O(n) time and O(1) auxiliary space;
validation, CDF computation and O(n log n) sample sorting precede it. No adjacency
lists are allocated. Strict distance comparisons and the h=0 convention are retained.

New regressions compare compiled and Python kernel evaluations, check actual
nopython dispatch for every criterion, verify input immutability, and enumerate
graphs at exact threshold distances and their floating-point neighbors. Existing
independent mathematical references continue to pass. No fastmath approximation
is enabled; different summation orders can still change final rounding bits.

A local warmed run on 2,000 Gamma observations (NumPy RNG seed 2026, shape 2.5,
rate 1.5) measured:

| Operation | Previous implementation | Optimized | Ratio |
|---|---:|---:|---:|
| Graph edge count, full validation/CDF/statistic call | 82.805 ms | 0.223 ms | 372× |
| Greenwood kernel, precomputed sorted CDF | 7.673 µs | 1.969 µs | 3.9× |

The graph baseline used the former generic adjacency-list path on the same
transformed sample; the Greenwood baseline used NumPy differences and squared
spacings. Results were checked for equality (graph) or numerical agreement
(Greenwood). Timings are minima of five repeats, with one old graph call, 100 new
graph calls and 1,000 Greenwood calls per repeat, divided by the call count.
JIT compilation was warmed before timing. These are local measurements, not
portable performance guarantees; the first invocation also pays compilation or
cache-loading cost. Greenwood's ratio excludes SciPy CDF computation.

### Reproducing the focused run

Ordinary collection is currently blocked by an unrelated obsolete
`AbstractChi2Statistic` import in `exponentiated_weibull.py`, reached through
`continuous/__init__.py`. The following explicit isolation bypasses **only that
package initializer**; it imports the actual Gamma implementation, factories and
kernels. It does not establish that normal package initialization works.
Run from the repository root:

```bash
.venv/bin/python - <<'PY'
import sys
import types
from pathlib import Path
import pytest

root = Path.cwd()
sys.path[:0] = [str(root), str(root / 'src')]
name = 'pysatl_criterion.statistics.goodness_of_fit.continuous'
package = types.ModuleType(name)
package.__path__ = [str(root / 'src/pysatl_criterion/statistics/goodness_of_fit/continuous')]
sys.modules[name] = package
raise SystemExit(pytest.main([
    '-q', 'tests/statistics/test_gamma_numba.py',
    'tests/statistics/test_gamma_dynamic.py',
    'tests/distributions/test_gamma.py',
    'tests/distributions/test_gamma_correctness.py',
]))
PY
```
