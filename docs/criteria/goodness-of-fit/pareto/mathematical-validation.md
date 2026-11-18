# Mathematical verification of Pareto I statistics

Audit scope: **21 concrete criteria** exported by the `pareto` package (`pareto/pareto_statistics.py`): 13 generated EDF
criteria, five generated frequency criteria and three composite-hypothesis
criteria. The [catalog](index.md) links every public class to its own page,
formula, executable example and scientific source.

## Null hypotheses and transformations

The model has zero location and CDF $F_{a,s}(x)=1-(s/x)^a$, $x\ge s>0$, $a>0$.
The generated criteria require **both parameters fixed**; Obradović requires
only scale fixed; Greenwood and KL require neither fixed. These are distinct
hypotheses, declared by the concrete classes rather than the family base.

For the specified model, the probability integral transform gives IID
$u_i=F_{a,s}(x_i)\sim U(0,1)$. Applying a fixed-CDF EDF functional is therefore
mathematically valid for Pareto without a separate Pareto-specific formula.
This argument does not permit estimating parameters and using the same tables.
For sorted values define $D^+=\max_i(i/n-u_i)$, $D^-=\max_i(u_i-(i-1)/n)$,
$r_i=i-1/2$, and $W^2=1/(12n)+\sum_i(u_i-r_i/n)^2$.

## Formula checks

All formulas below agree with the implementations on the checked domain.
The source column distinguishes original formulas from adaptations.

| Criterion | Checked returned value | Scientific source |
|---|---|---|
| Kolmogorov–Smirnov D+ | $D^+$, no scaling | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256); [Zhang (2002), §2.1](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf) |
| Kolmogorov–Smirnov D− | $D^-$, no scaling | Same sources |
| Kolmogorov–Smirnov | $\max(D^+,D^-)$, not squared | [Zhang (2002), §2.1](https://doi.org/10.1111/1467-9868.00337) |
| Anderson–Darling | $-n-n^{-1}\sum_i(2i-1)[\log u_i+\log(1-u_{n+1-i})]$ | [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437); Zhang §2.2 |
| Cramér–von Mises | $W^2$ | Zhang §2.3 |
| Liao–Shimokawa | $n^{-1/2}\sum_i\max(i/n-u_i,u_i-(i-1)/n)/\sqrt{u_i(1-u_i)}$ | [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965); fixed-CDF adaptation, see qualifications below |
| Watson | $W^2-n(\bar u-1/2)^2$ | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109); [Lemeshko et al. (2014), eq. (6)](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf) |
| Kuiper | $D^++D^-$ | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60)50006-0); Lemeshko eq. (1) |
| Scaled Kuiper | $\sqrt n(D^++D^-)$ | Lemeshko, discussion before eq. (2) |
| Corrected Kuiper | $(\sqrt n+0.155+0.24/\sqrt n)(D^++D^-)$ | Lemeshko eq. (3); the multiplier approximates finite-sample stabilization |
| Zhang K | $\max_i[r_i\log(r_i/(nu_i))+(n-r_i)\log((n-r_i)/(n(1-u_i)))]$ | Zhang eq. (3.1) |
| Zhang A | $-\sum_i[\log u_i/(n-r_i)+\log(1-u_i)/r_i]$ | Zhang eq. (3.2) |
| Zhang C | $\sum_i[\log((1-u_i)/u_i)-\log((n-i+1/4)/(i-3/4))]^2$ | Zhang eq. (3.3) |
| Pearson chi-square | $\sum_j(O_j-E_j)^2/E_j$ | [Pearson (1900)](https://doi.org/10.1080/14786440009463897); [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), $\lambda=1$ |
| Freeman–Tukey | $4\sum_j(\sqrt{O_j}-\sqrt{E_j})^2$ | Cressie–Read, $\lambda=-1/2$ |
| Modified log-likelihood | $2\sum_jE_j\log(E_j/O_j)$ | Cressie–Read, $\lambda=-1$ |
| Neyman modified chi-square | $\sum_j(O_j-E_j)^2/O_j$ | Cressie–Read, $\lambda=-2$ |
| Cressie–Read | $(9/5)\sum_jO_j[(O_j/E_j)^{2/3}-1]$ | Cressie–Read, $\lambda=2/3$ |
| Obradović | $n^{-1}\sum_i[M_n(z_i)-F_n(z_i)]$, $z_i=x_i/s$ | [Obradović et al. (2014), §2](https://arxiv.org/html/1310.5510), signed integral $T_n$ |
| Greenwood | $\sum_i\log^2(x_i/x_{(1)})/[\sum_i\log(x_i/x_{(1)})]^2$ | Library log-residual adaptation of the concentration functional in [Does et al. (1988)](https://ir.cwi.nl/pub/1694/1694D.pdf); derivation below |
| Lequesne KL | $-V_{mn}-n^{-1}\sum_i\log f_{\widehat a,\widehat s}(x_i)$ | [Lequesne and Regnault (2020)](https://doi.org/10.18637/jss.v096.c01); [preprint eqs. (5), (8)](https://arxiv.org/html/1806.07244), fixed-window specialization |

The frequency kernels require exhaustive cells with positive expected counts
and matching totals. With equal-probability binning, $E_j=n/k$; explicit edges
use their null probability differences. The coefficients and zero-count limits
were checked independently. For $O_j=0<E_j$, modified log-likelihood and Neyman
are infinite, Freeman–Tukey contributes $4E_j$, Pearson contributes $E_j$, and
Cressie–Read contributes zero. Returning infinity in these cases is intentional.

## Composite-hypothesis derivations

### Obradović

The pairwise empirical CDF uses each unordered pair once and denominator
$\binom n2$. Integrating it against $F_n$ is exactly its average over the sample,
which accounts for the additional $1/n$ factor. The implementation subtracts
$F_n(z_i)$ with inclusive ties, rather than replacing it by a no-tie rank formula.
A positive power of all $z_i$ preserves the pairwise comparisons, so nuisance
shape disappears. Known scale is removed by division. The upper-tail convention
and signed statistic follow §2; no absolute value or studentization is inserted.

### Greenwood

This is a direct derivation for the implementation, not an assertion that the
cited spacing paper defines a fitted-Pareto test. Since $\log(X/s)$ is exponential,
subtracting its sample minimum leaves one zero and $k=n-1$ memoryless exponential
residuals. Their normalized proportions have a $k$-component Dirichlet law.
Its moments give

$$
\mathbb EG=\frac2{k+1}=\frac2n,\quad
\operatorname{Var}(G)=\frac{4(k-1)}{(k+1)^2(k+2)(k+3)}
=\frac{4(n-2)}{n^2(n+1)(n+2)}.
$$

Cauchy–Schwarz yields $1/(n-1)\le G\le1$. The ratio is unchanged by multiplying
all observations by a positive constant or raising them to a positive power.
At $n=2$ it is identically one; the implementation requires $n\ge3$ and a
nonconstant sample. Both tails are used. Calibration for ordinary $n+1$ uniform
CDF spacings must not be used without adjusting the number of components.

### Fixed-window KL

Maximizing the Pareto likelihood gives $\widehat s=x_{(1)}$ and
$\widehat a=1/\bar y$ with $y_i=\log(x_i/\widehat s)$.
Substitution into the fitted mean negative log density gives
$\log\bar y+\log\widehat s+1+\bar y$. Subtracting $V_{mn}$ gives the implemented
expression. Clamped endpoint indices and the factor $n/(2m)$ match the Vasicek
entropy estimator. Scaling the data adds the same logarithm to entropy and
negative log density, hence cancels. A power transformation does not cancel
in the finite-sample spacing estimator, so shape-dependent calibration remains.

## Scope, sources and qualifications

This audit verifies statistic calculations and their declared hypotheses. It
does not establish exact rejection probabilities, optimality, or power against
all alternatives. Built-in Pareto simulation remains unavailable. External
calibration must reproduce sample size, the selected statistic normalization,
binning, parameter fitting, window selection and any measurement rounding.

Full text was checked for Zhang, Lemeshko et al., Obradović et al., Lequesne and
Regnault, and Does et al. The Cressie–Read definition was checked in the
[article text hosted by NYU](https://pages.stern.nyu.edu/~dbackus/Exotic/1Robustness/CressieRead%20JRSSB%2084.pdf).
Original Liao–Shimokawa full text could not be retrieved; its expression was
checked in [Al-Zahrani (2012), eq. (13)](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf).
The latter studies Topp–Leone distributions. Neither paper supplies calibration
for this library's fixed-Pareto adaptation. The criterion uses the shared Liao–Shimokawa kernel through the EDF factory.

The Greenwood source supports the squared-spacing functional, not this exact
Pareto construction. Its null derivation is supplied above. KL is the raw
fixed-window estimate, not the adaptive-window, entropy-constrained, bias-corrected
`vsgoftest` workflow. Negative KL estimates are possible and are not clipped.
The default square-root window does not by itself justify the original paper's
asymptotic normal p-value approximation. Plug-in shape bootstrap is approximate;
generic stored KL calibration is deliberately rejected.

Small-sample degeneracy is explicit: Watson is $1/12$ at $n=1$; raw and scaled
Kuiper are one and corrected Kuiper is 1.395. These calculations are defined but
uninformative. Corrected Kuiper's asymptotic calibration is not exact at small $n$.

## Numerical verification and corrections

The focused checks include:

- All 13 EDF statistics against independent formulas on four shape/scale pairs;
  AD, CVM and Watson also against numerical EDF integrals, and KS against SciPy.
- AD, the weighted Liao–Shimokawa functional and all three Zhang statistics
  against 240-digit `decimal` references with upper-tail CDFs rounding to one.
- All five frequency statistics against SciPy power divergences or their
  analytic zero-count limits, using explicit and equal-probability cells.
- Greenwood's exact ratio, scale/power invariance, range and Dirichlet moments
  in deterministic seeded simulation at $n=3,7,20$ (6,000 replicates each).
- Obradović's signed sum against brute-force inclusive indicators and exact
  rational comparisons, including tied values and overflowing ordinary ratios.
- KL against independent fitted Pareto log densities and SciPy's Vasicek entropy,
  across windows; scale invariance, lack of shape invariance, negative values,
  tied zero spacings and invalid windows.
- Hypothesis declarations, invalid data, lower endpoints, adjacent large floats,
  no mutation or retained fits, masked input, public stubs and documentation.

No mathematical formula changes were needed. The audit found and fixed an input
validation defect: the three explicit composite criteria silently discarded
`MaskedArray` masks and used hidden observations. They now reject masked input,
consistent with the generated criteria. Regression tests cover all 21 classes.

The focused suite passed **527 tests**. It includes documentation coverage for
all 21 classes, execution of every criterion-page and overview example, local
link validation and checks for scientific references and examples in docstrings.
Ruff and the `pareto/__init__.pyi` generator's `--check` passed.
MkDocs built successfully; its warnings concern existing broken links in other
families, with no Pareto page warnings.

Ordinary pytest collection is still blocked by the existing import of the removed
`AbstractChi2Statistic` from `edf.py` in `exponentiated_weibull.py`. The successful
run bypassed only `continuous/__init__.py`, loading the actual Pareto module,
shared kernels, binning and distribution implementation. This does **not** verify
normal package initialization or resolve the unrelated migration errors.

From the repository root, the isolated focused run is reproducible with:

```sh
.venv/bin/python - <<'PY'
import sys
import types
from pathlib import Path
import pytest

name = "pysatl_criterion.statistics.goodness_of_fit.continuous"
package = types.ModuleType(name)
package.__path__ = [str(Path("src/pysatl_criterion/statistics/goodness_of_fit/continuous").resolve())]
sys.modules[name] = package
raise SystemExit(pytest.main([
    "-q",
    "tests/distributions/test_pareto.py",
    "tests/distributions/test_pareto_correctness.py",
    "tests/statistics/test_pareto_dynamic.py",
    "tests/statistics/test_pareto_mathematical_validation.py",
    "tests/statistics/test_pareto_documentation.py",
    "tests/statistics/test_pareto_numba.py",
]))
PY
.venv/bin/python scripts/generate_pareto_stub.py --check
```

After the obsolete imports are migrated, the same six test paths can be supplied
to ordinary `python -m pytest -q` without the bootstrap.

## Numba optimization

All 21 criteria execute compiled numerical kernels. The 18 generated EDF and
frequency criteria reuse the shared Numba kernels. Greenwood and fixed-window
KL now have static `do_execute_statistic` kernels decorated with
`@jit(nopython=True, cache=True)`. Stable log-ratio and log-probability evaluation
also use cached nopython methods. Validation, hypothesis handling and all helper
methods reside inside the Pareto classes; there are no module-level helper functions.
No `fastmath` approximation is enabled.

Obradović uses a compiled pair-counting kernel on unambiguous data. If its
logarithmic comparison tolerance detects a potentially ambiguous boundary,
the kernel returns an internal NaN sentinel and the public method recomputes
using exact `Fraction` comparisons. The sentinel is never returned as the
statistic. This preserves inclusive ties, extreme ratios and the original
binary-float semantics. Exact arithmetic remains in Python; the whole public
method is not claimed to be nopython. Worst-case complexity and memory bounds
are unchanged.

`test_pareto_numba.py` verifies compiled public execution paths for every class,
the class-only helper layout, fast-path execution without the exact fallback,
and agreement with rational references when fallback is needed. The existing
high-precision, boundary, invariance and documentation checks remain in the suite.

A local warmed benchmark compared the pre-optimization implementation with the
new public `execute_statistic` calls, including validation and sorting. It used
seed 2026, samples `1 + rng.pareto(2, n)`, the minimum of five repeated timings,
1,000 calls per repeat except 10 for Obradović. Compilation was excluded.
These are local measurements, not performance guarantees:

| Criterion | Sample size | Before, microseconds/call | After, microseconds/call | Speedup |
|---|---:|---:|---:|---:|
| Greenwood | 1,000 | 40.0 | 30.3 | 1.32× |
| Fixed-window KL | 1,000 | 53.0 | 34.1 | 1.55× |
| Obradović, ordinary continuous sample | 500 | 10,393.9 | 4,273.8 | 2.43× |
| Anderson–Darling | 1,000 | 75.0 | 61.0 | 1.23× |

First use incurs compilation or cache-loading cost. Samples requiring rational
fallback may not obtain the ordinary-sample Obradović speedup.

## Package layout

The implementation follows the `continuous/uniform` layout:

```text
continuous/pareto/
    __init__.py
    __init__.pyi
    pareto_statistics.py
```

The package initializer explicitly re-exports all public classes and preserves
`pysatl_criterion.statistics.goodness_of_fit.continuous.pareto` as their public
module path. Existing imports and pickled class references remain valid.
Private factory catalogs live in `pareto_statistics.py`. The stub generator reads
that implementation and writes `pareto/__init__.pyi`; use its `--check` option to
verify freshness. Tests check package/implementation class identity and old pickle
references for every exported class, and execute class docstrings using the
implementation module's globals.
