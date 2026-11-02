# Mathematical verification of exponentiated Weibull statistics

The [18 public criteria](index.md) were audited: **13 EDF statistics and five
frequency statistics**. This report follows the structure of the
[uniformity audit](../uniformity/mathematical-validation.md).

## Hypothesis and scope

The independent observations have the fully specified null

$$
F_{a,k,s}(x)=[1-\exp(-(x/s)^k)]^a,\quad x\ge0,\qquad a,k,s>0.
$$

The location is zero. `exponent`, `shape` and `scale` must all be fixed before
observing the data; partial parameter sets are rejected. The exponent is not the
ordinary Weibull shape. At $a=1$ this distribution reduces to ordinary Weibull.
The parameterization agrees with [SciPy's `exponweib`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html)
(`a=exponent`, `c=shape`, `scale=scale`, `loc=0`).

Write $u_i=F_{a,k,s}(x_{(i)})$, $r_i=i-1/2$, and
$D^+=\max_i(i/n-u_i)$, $D^-=\max_i(u_i-(i-1)/n)$.
The probability integral transform supplies the fixed uniform null. All classes
return statistics, not p-values; all use right-tail rejection, including D+ and D−.
No parameter fitting or fitted-Weibull critical-value correction is performed.

## Formula checks

| Criterion | Verified definition or normalization | Scientific source |
|---|---|---|
| Kolmogorov–Smirnov D+ | $D^+$; no square or $\sqrt n$ factor | Zhang (2002), §2.1 |
| Kolmogorov–Smirnov D− | $D^-$; right-tail rejection | Zhang (2002), §2.1 |
| Kolmogorov–Smirnov | $\max(D^+,D^-)$; legacy direction options select the same directional statistics | Zhang (2002), §2.1 |
| Anderson–Darling | $-n-n^{-1}\sum_i(2i-1)[\log u_i+\log(1-u_{n+1-i})]$ | Zhang (2002), §2.2 |
| Cramér–von Mises | $1/(12n)+\sum_i(u_i-r_i/n)^2$ | Zhang (2002), §2.3 |
| Liao–Shimokawa | $n^{-1/2}\sum_i\max(i/n-u_i,u_i-(i-1)/n)/\sqrt{u_i(1-u_i)}$ | Liao and Shimokawa (1999); cross-check in Al-Zahrani (2012), eq. (13) |
| Watson | $W^2-n(\bar u-1/2)^2$, computed as centered squared deviations | Lemeshko et al. (2014), eq. (6); Watson (1961) |
| Kuiper | $V=D^++D^-$ | Lemeshko et al. (2014), eq. (1) |
| Scaled Kuiper | $\sqrt n V$ | Lemeshko et al. (2014), discussion before eq. (2) |
| Corrected Kuiper | $(\sqrt n+0.155+0.24/\sqrt n)V$ | Lemeshko et al. (2014), eq. (3) |
| Zhang K | $\max_i[r_i\log(r_i/(nu_i))+(n-r_i)\log((n-r_i)/(n(1-u_i)))]$ | Zhang (2002), eq. (3.1) |
| Zhang A | $-\sum_i[\log u_i/(n-r_i)+\log(1-u_i)/r_i]$ | Zhang (2002), eq. (3.2) |
| Zhang C | $\sum_i[\log((1-u_i)/u_i)-\log((n-i+1/4)/(i-3/4))]^2$ | Zhang (2002), eq. (3.3) |
| Pearson | $\sum_j(O_j-E_j)^2/E_j$, power divergence $\lambda=1$ | Cressie and Read (1984) |
| Freeman–Tukey | $4\sum_j(\sqrt{O_j}-\sqrt{E_j})^2$, $\lambda=-1/2$ | Cressie and Read (1984) |
| Modified log-likelihood | $2\sum_jE_j\log(E_j/O_j)$, $\lambda=-1$ | Cressie and Read (1984) |
| Neyman modified chi-squared | $\sum_j(O_j-E_j)^2/O_j$, $\lambda=-2$ | Cressie and Read (1984) |
| Cressie–Read | $(9/5)\sum_jO_j[(O_j/E_j)^{2/3}-1]$, $\lambda=2/3$ | Cressie and Read (1984) |

The shared EDF and divergence kernels match these definitions. The audit found
numerical errors in the family probability calculations, rather than incorrect
normalizations in these kernels; the corrections are described below.

## Sources and verification limits

- [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), [full text](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf).
- [Lemeshko et al. (2014), Solving Problems of Using Some Nonparametric Goodness-of-Fit Tests](https://doi.org/10.3103/S875669901401004X), [authors' university copy](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf).
- [Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x), [full text](https://pages.stern.nyu.edu/~dbackus/Exotic/1Robustness/CressieRead%20JRSSB%2084.pdf).
- [Liao and Shimokawa (1999), A new goodness-of-fit test for type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965).
- [Al-Zahrani (2012), Goodness-of-Fit for the Topp-Leone Distribution with Unknown Parameters](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf), eq. (13).
- [Watson (1961), Goodness-of-fit tests on a circle](https://doi.org/10.1093/biomet/48.1-2.109).

Zhang and Lemeshko were checked in accessible full texts. The power-divergence
family and its parameter correspondences were checked against the accessible
text of Cressie and Read. The original Liao–Shimokawa full text was unavailable:
its bibliographic record and abstract were checked, and the implemented weighted
EDF expression was cross-checked in Al-Zahrani's application. That latter paper
is not used as a reference for the other EDF formulas or for exponentiated
Weibull critical values.

The original Liao–Shimokawa procedure estimates parameters, whereas this family uses
a specified CDF. Its fitted-null tables are therefore unsuitable here.
Zhang C implements the published quarter-rank approximation in eq. (3.3), not
the preceding integrated statistic with its constants. Corrected Kuiper uses
an empirical normalization, not an exact finite-sample distribution.

## Numerical corrections

1. **False infinite logarithmic penalties for tiny exponents.** For
   `exponent=1e-320`, `shape=scale=1` and sample `[1, 2, 10]`, the old
   Anderson–Darling calculation returned infinity. An 800-digit Decimal
   evaluation gives approximately **2214.0422443729744**. Multiplication by
   the exponent rounded $\log F$ to a subnormal value or zero before computing
   $\log(1-F)$. The family probability kernel now computes
   $\log q=\log a+\log[-\log(1-e^{-t})]$, where $t=(x/s)^k$ and
   $q=-\log F$, and uses $\log(1-e^{-q})\simeq\log q$ for tiny $q$.
   The regression also covers Liao–Shimokawa and all three Zhang statistics.
2. **Overflow before a compensating exponent.** For `shape=1e308`,
   `exponent=1e-308`, `scale=1` and positive observations below one,
   $k\log(x/s)$ can overflow negatively even though $ak\log(x/s)$ is finite.
   The lower-tail branch now evaluates the latter product directly. The
   limiting CDF in the regression is $F(x)\simeq x$; KS is checked against
   an independent uniform reference.
3. **Frequency cells computed from an underflowed intermediate CDF.** At
   `shape=1000`, `exponent=0.001`, `scale=1`, directly evaluating the ordinary
   Weibull CDF before exponentiation can round it to zero. For sample
   `[0.01, 0.1, 0.4, 0.8]`, the correct two-cell Pearson value is **1**.
   Frequency binning now uses the same stable CDF and logarithmic tails as
   EDF evaluation. Both equal-probability and explicit-edge bins are checked.

These are floating-point corrections to the specified mathematical model.
Exact zero observations still give infinite logarithmic EDF penalties.
Statistics whose true magnitudes exceed float64 can still be infinite.
Extreme parameter combinations may also be unsuitable for sampling with the
SciPy-backed generator; this audit does not certify that generator across the
entire floating-point parameter range.

## Frequency bins and calibration

The current default is
`EqualProbabilityBins(PowerCount(2.0, 0.4), min_expected=None, max_bins=1000)`:

$$
B=\min(1000,\max(2,\lceil2n^{0.4}\rceil)),\qquad E_j=n/B.
$$

The previous Pearson documentation incorrectly retained $\lceil\sqrt n\rceil$
and a one-cell singleton. The pages now describe the actual default, its cap,
and explicit binning options. At $n=1$ the default has two cells: Pearson is 1,
not 0. Zero observed cells produce infinity for modified log-likelihood and
Neyman; Freeman–Tukey, Pearson and Cressie–Read have finite limits.

For fixed exhaustive cells with positive probabilities, the large-sample
chi-squared reference has $B-1$ degrees of freedom because no parameters are
estimated. This does not establish accuracy with small expected counts or
sample-size-dependent bins. Monte Carlo must reproduce the chosen statistic,
sample size and binning. The family still rejects stored unversioned critical
values. Changing the binning or statistic identifier requires recalibration;
the current Watson short code is `WATSON`, not the historical `W`.

## Package and type declarations

The public package follows the uniform family layout:

```text
continuous/exponentiated_weibull/
    __init__.py
    __init__.pyi
    exponentiated_weibull_statistics.py
```

`__init__.py` re-exports all 19 public classes (one abstract family base and
18 criteria). The public import paths and class module identifiers remain
unchanged, including compatibility with existing pickle references.
`__init__.pyi` describes the generated classes, per-criterion hypotheses,
frequency `binning` arguments, and the legacy KS direction and mode options.
Regenerate and check it without importing the package:

```bash
python scripts/generate_exponentiated_weibull_stub.py
python scripts/generate_exponentiated_weibull_stub.py --check
```

## Numba implementation

All 18 criteria use the compiled shared EDF or frequency statistic kernels.
`AbstractExponentiatedWeibullGofStatistic.do_log_probabilities` additionally uses
`@jit(nopython=True, cache=True)` to evaluate the family probabilities in one
pass. It avoids intermediate masks and arrays, and computes survival logs only
when required. EDF criteria without logarithmic kernels and frequency CDF
binning use the CDF-only path. Validation, sorting, strategy selection and
parameter objects remain outside the compiled kernel, as in the uniform family.

All family callbacks are class methods; the module has no module-level function
definitions. No `fastmath` is enabled: NaNs, infinities and the tail safeguards
above retain their intended meaning. Compilation has a first-call cost; timings
of repeated evaluations must exclude compilation. Tests check nopython dispatch
for every criterion and compare compiled results with the Python kernel at
ordinary and extreme parameter values.

## Numerical verification and documentation coverage

The dedicated suite is
`tests/statistics/test_exponentiated_weibull_mathematical_validation.py`.
It checks all 18 formulas using independent EDF quadrature, SciPy KS and
multinomial power divergences, direct rank formulas, and 800-digit Decimal
references. It includes zero observations, ties, singletons, empty cells,
legacy KS directions, extreme exponents, intermediate overflow, and the
probability adapter's scalar/array and support contracts.

Every criterion has a dedicated page, explicit formula, scientific links and
an executable example. The coverage test matches pages against the actual
runtime catalog, checks the index and MkDocs navigation, and executes all 18
examples. Public class docstrings also contain examples and scientific links.

The focused isolated run passed **275 tests**, including execution of all 18
page examples and all 72 docstring example steps. Ruff passed for the changed
Python files. MkDocs built successfully; existing warnings concern other
sections, with no warnings for exponentiated Weibull pages.

The ordinary pytest invocation is currently blocked during package import by
an unrelated obsolete `AbstractChi2Statistic` import in `inverse_gamma.py`.
The focused run bypasses only the eager `continuous/__init__.py` imports; it
loads the real family, factories, kernels, distributions and binning code.
Consequently the results verify these calculations and documentation, but do
not establish that ordinary package initialization works.

A reproducible isolated invocation from the repository root is:

```python
import sys
import types
from pathlib import Path
import pytest

name = "pysatl_criterion.statistics.goodness_of_fit.continuous"
package = types.ModuleType(name)
package.__path__ = [
    str(Path("src/pysatl_criterion/statistics/goodness_of_fit/continuous").resolve())
]
sys.modules[name] = package
raise SystemExit(
    pytest.main(
        [
            "tests/statistics/test_exponentiated_weibull_mathematical_validation.py",
            "tests/statistics/test_exponentiated_weibull_dynamic.py",
            "tests/distributions/test_weibull_families.py",
            "tests/statistics/test_exponentiated_weibull_numba.py",
            "-q",
        ]
    )
)
```

The checks establish agreement with the stated formulas on the tested cases.
They are not a proof of exact rejection probabilities, a power study, or a
certification of all possible float64 parameter combinations.
