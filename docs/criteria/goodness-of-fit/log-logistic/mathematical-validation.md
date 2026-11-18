# Mathematical verification of log-logistic statistics

Audit date: 2026-10-17. All **20 concrete criteria** in `log_logistic/log_logistic_statistics.py` were
reviewed: 13 generated EDF statistics, 5 generated frequency statistics and the
2 fitted-model statistics Nikulin and Mirvaliev. The report follows the
[uniformity verification report](../uniformity/mathematical-validation.md).
The [catalog](index.md) links to an individual page for every criterion.

## Formula checks

Write $u_i=F_{\alpha,\beta}(x_{(i)})$ and $r_i=i-1/2$.
For positive observations,

$$
z_i=\beta(\log x_i-\log\alpha),\qquad
F(x_i)=\operatorname{expit}(z_i),\quad
\log F(x_i)=-\log(1+e^{-z_i}),\quad
\log S(x_i)=-\log(1+e^{z_i}).
$$

The implementation uses `logaddexp` for the last two expressions. The generated
criteria require both parameters fixed. The two bespoke criteria require both
unknown; hypothesis support is declared separately by each concrete criterion.

| Criteria (number) | Checked definition | Scientific source |
|---|---|---|
| KS D+, D−, D (3) | $D^+=\max(i/n-u_i)$, $D^-=\max(u_i-(i-1)/n)$, $D=\max(D^+,D^-)$; no square or scaling | [Zhang (2002), §2.1](https://doi.org/10.1111/1467-9868.00337) |
| Anderson–Darling (1) | Weighted logarithmic sum, factor $1/n$, subtraction of $n$ | [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437); [Zhang (2002), §2.2](https://doi.org/10.1111/1467-9868.00337) |
| Cramér–von Mises (1) | $1/(12n)+\sum(u_i-r_i/n)^2$ | [Zhang (2002), §2.3](https://doi.org/10.1111/1467-9868.00337) |
| Watson (1) | $W^2-n(\bar u-1/2)^2$, evaluated with centered deviations | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109); [Lemeshko et al. (2014), eq. (6)](https://doi.org/10.3103/S875669901401004X) |
| Kuiper, scaled, corrected (3) | $V=D^++D^-$, $\sqrt nV$, $(\sqrt n+0.155+0.24/\sqrt n)V$ | [Lemeshko et al. (2014), eqs. (1)–(3)](https://doi.org/10.3103/S875669901401004X) |
| Zhang K, A, C (3) | Half-rank likelihood ratios and quarter-rank log-odds correction | [Zhang (2002), eqs. (3.1)–(3.3)](https://doi.org/10.1111/1467-9868.00337) |
| Liao–Shimokawa (1) | $n^{-1/2}\sum_i\max(i/n-u_i,u_i-(i-1)/n)/\sqrt{u_i(1-u_i)}$ | [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965); [Al-Zahrani (2012), eq. (13)](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf) |
| Pearson, Freeman–Tukey, modified log-likelihood, Neyman, Cressie–Read (5) | Power-divergence family at $\lambda=1,-1/2,-1,-2,2/3$; zero-count limits checked explicitly | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| Bagdonavicius–Nikulin (1) | Exposure residuals and information correction with an observed-count denominator | [Bagdonavicius and Nikulin (2011), §§3–5](https://www.numdam.org/item/10.1016/j.crma.2011.01.007.pdf) |
| Mirvaliev (1) | Moment-fitted logistic quadratic form applied to $\log X$ | [Pya (2004), pp. 70–71, eqs. (8)–(15)](https://math.kz/media/journal/journal2017-06-0725590.pdf#page=70) |

No algebraic changes were needed to the statistic kernels checked here.
All criteria reject in the right tail; directional KS refers to CDF deviation,
not to the rejection tail of the returned nonnegative statistic.

### Bagdonavicius–Nikulin derivation

With $z=(\log t-\mu)/s$, the hazard is $h(t)=F(t)/(st)$ and cumulative
hazard is $H(t)=\log(1+e^z)$. The implemented hazard-score coordinates are
$(-S,-1-zS)$, an invertible rescaling of the location/log-scale score at the fit.
Consequently the quadratic form is unchanged by this coordinate choice.

Boundaries equalize $\sum_i\min(H(t_i),H(a_j))$. With event counts $U_j$,
exposures $e_j$, $Z=(U-e)/\sqrt n$, $A=\operatorname{diag}(U/n)$, cell score
sums $C$ divided by $n$, and event score outer products $I$ divided by $n$,
the implementation returns

$$
Z^TA^{-1}Z+(CA^{-1}Z)^T(I-CA^{-1}C^T)^{-1}(CA^{-1}Z).
$$

Within-cell centering computes the latter matrix without cancellation.
An independent reference uses SciPy's censored logistic likelihood fit,
numerically integrates exposure over the risk sets, and inverts
$A-C^TI^{-1}C$ directly. Agreement verifies both the coordinate change and
Woodbury rearrangement. Empty event cells return infinity; singular or
ill-conditioned correction matrices raise an error instead of being regularized.

### Mirvaliev derivation

Standardized logs use the sample variance with divisor $n$. For the unit-variance
logistic null, $E[Y^4]=21/5$, hence
$V=\operatorname{Cov}(Y,Y^2-1)=\operatorname{diag}(1,16/5)$ and
$K=\operatorname{diag}(1,2)$. The implemented matrices give

$$
A=I-qq^T+C(V-C^TC)^{-1}C^T,\quad D=C-BK^{-1}V,\quad L=V+D^TAD,
$$

and $v^TAv-(D^TAv)^TL^{-1}(D^TAv)$ for standardized cell residuals $v$.
This reproduces Pya's moment-based correction in standardized coordinates.
An independent check constructs the covariance of fitted cell residuals from
integrals of the mean/standard-deviation influence functions and evaluates its
pseudoinverse quadratic form. No normal-distribution fourth moment is substituted.

## Numerical correction found during the audit

The shared distribution adapter previously delegated log-logistic probabilities
to SciPy's Fisk evaluator, which could underflow the ratio $x/\alpha$ before
applying a small shape exponent. For `alpha=1e250`, `beta=0.001`, sample
`[1e-40, 1e-60, 1e-100, 1e-250]` and ten equal-probability cells, Pearson
returned **16 instead of 21**. Freeman–Tukey and Cressie–Read were also affected.
The problem was in cell assignment, not their discrepancy formulas.

The log-logistic bound distribution now evaluates CDF, SF, log-CDF and log-SF
from log odds. It retains support limits and NaN propagation. This also repairs
premature infinite log probabilities: at `alpha=1`, `beta=2`, `x=1e-250`,
`logcdf` is approximately −1151.292546497, not negative infinity.
EDF kernels already used stable log odds and did not require this correction.
Old calibrations or results computed in the affected numerical regimes should
be regenerated; ordinary-range definitions and normalizations are unchanged.

## Sources and qualifications

Full text was inspected for [Zhang](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf),
[Lemeshko et al.](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf),
[Bagdonavicius–Nikulin](https://www.numdam.org/item/10.1016/j.crma.2011.01.007.pdf)
and [Pya](https://math.kz/media/journal/journal2017-06-0725590.pdf).
Cressie–Read's publisher summary identifies the five divergence parameters;
their formulas and limiting zero-count contributions were additionally checked
algebraically and against SciPy.

The original Liao–Shimokawa full text was inaccessible in this audit. Its
bibliography and fitted-model scope were checked, and the expression was
compared with its use in Al-Zahrani's published study. The implementation uses
a **local maximum of two deviations at each observation**. This is a fixed-null
log-logistic adaptation; original fitted-Weibull critical values do not apply.
The original Mirvaliev (2001) paper was not inspected: the logistic application
is checked against Pya (2004), then transferred through the logarithm.

These checks establish agreement of the calculations with the stated definitions
on the tested domains. They do not prove finite-sample significance levels,
omnibus consistency of every discrepancy, or power against every alternative.
Ties are computable but continuous-null critical values are not automatically
valid for rounded data. Frequency chi-square approximations require appropriate
cell counts and assumptions. Neither default bin count nor Kuiper correction
makes an asymptotic reference exact.

Nikulin uses the sample maximum as the observation horizon. The source assumes
an observation scheme and regularity conditions; no exact finite-sample law for
this random horizon is asserted. External calibration must reproduce censoring,
observation/stopping, fitting and grouping. Mirvaliev supports only complete data
and must refit moments in simulation. The package has no registered log-logistic
random generator; the generic Monte Carlo resolver is therefore unavailable.

## Numerical verification and documentation coverage

The focused suite passed **230 tests** across:

- `tests/distributions/test_log_logistic.py`;
- `tests/distributions/test_log_logistic_correctness.py`;
- `tests/distributions/test_log_logistic_mathematical_validation.py`.

References include EDF integrals (AD, CvM, Watson), SciPy KS and multinomial
power divergences, 250-digit Decimal tail calculations, independent fitted-model
covariance expressions, censored likelihood fits and risk integrals. Checks
cover all 20 criteria, fixed/unknown parameter contracts, transformations,
unsorted input, ties, singletons, invalid data, empty cells, extreme scales,
nonmutation and calibration restrictions.

The documentation check finds exactly one page per concrete class, verifies its
catalog and navigation entry and scientific URL, and executes every individual
page's Python example. Generated class docstrings also have executable examples.
There are **20 criterion pages, one catalog and this report**; six existing
criterion URLs are preserved. Formula, hypothesis, input, rejection direction,
binning and calibration information are included.

Ordinary test collection is currently blocked by an unrelated pre-existing
`NameError: AbstractCrammerVonMisesStatistic` in `continuous/log_normal.py`.
The focused run supplies the `continuous` package path without executing its
aggregate `__init__.py`; the actual log-logistic module, factories, kernels and
probability adapter execute normally. Thus these results do not certify that
ordinary package initialization succeeds in the current working tree.

The shared probability-evaluation suite additionally passed **50 tests** after
the adapter correction. Ruff and whitespace checks passed for the changed code.
MkDocs built the site successfully; its six broken-link warnings concern
pre-existing Weibull audit links, not the log-logistic section.

To reproduce the isolated focused run from the repository root:

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
            "tests/distributions/test_log_logistic.py",
            "tests/distributions/test_log_logistic_correctness.py",
            "tests/distributions/test_log_logistic_mathematical_validation.py",
            "tests/distributions/test_distribution_evaluation.py",
            "-q",
        ]
    )
)
```

## Numba optimization follow-up

The 18 generated criteria retain the compiled shared EDF/frequency kernels.
Local numerical work now uses `@jit(nopython=True, cache=True)` for CDF and log
probabilities, Nikulin likelihood terms and exposure/information accumulation,
and Mirvaliev moment integrands, fitted cell residuals and the final quadratic
form. Validation, SciPy optimization/quadrature and checked matrix solves remain
Python orchestration. All helpers in `log_logistic/log_logistic_statistics.py` are class methods.

Mirvaliev caches only distribution-dependent edges, derivatives and integrals
by interval count (LRU bound: 32 entries); cached arrays are read-only. Sample
moments and residuals are recomputed for every call. Nikulin reduces likelihood
terms with NumPy to preserve its pairwise summation and optimizer convergence.
No fast-math mode or relaxed optimizer tolerance is used.

A local warm-cache measurement used 1,000 observations from
`exp(default_rng(13).logistic(size=1000))`, default criterion settings, and the
minimum of three batches of 20 calls. Compilation was excluded:

| Criterion | Before | After | Approximate speedup |
|---|---:|---:|---:|
| Anderson–Darling | 0.054 ms | 0.053 ms | No material change |
| Nikulin | 2.20 ms | 1.90 ms | 1.16× |
| Mirvaliev | 0.478 ms | 0.111 ms | 4.3× |

These timings describe this machine and sample, not a universal performance
bound. First-call compilation and uncached quadrature cost extra. Values agree
with the original implementation within floating-point precision on this
sample. The expanded focused suite passes **237 tests**, including independent
likelihood/finite-difference gradient checks, nopython compilation of every
local numerical kernel and all generated criteria, bounded/read-only cache
behavior, and absence of module-level helper functions. The ordinary-import
limitation noted above still applies.
