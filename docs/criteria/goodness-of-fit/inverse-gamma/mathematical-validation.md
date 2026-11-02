# Mathematical verification of inverse-gamma statistics

## Scope and conclusion

All **19 concrete public classes** in `continuous/inverse_gamma/inverse_gamma_statistics.py` were audited.
They implement 19 distinct scalar definitions.
No mathematical formula changes were needed for the definitions checked.
The complete class-to-page inventory is in the [overview](index.md).

This report follows the [uniformity verification report](../uniformity/mathematical-validation.md).
It verifies the implemented statistics and documented conventions, not the power
of each test against every alternative or exact finite-sample rejection rates.

## Null model and probability transform

The hypothesis is iid inverse gamma with fixed shape $\alpha>0$, scale $\beta>0$
and zero location:

$$
f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{-\alpha-1}e^{-\beta/x},\quad x>0.
$$

Setting $Y=\beta/X$ and applying the Jacobian gives
$f_Y(y)=y^{\alpha-1}e^{-y}/\Gamma(\alpha)$, hence
$F_X(x)=P(Y\ge\beta/x)=Q(\alpha,\beta/x)$.
This verifies that the CDF uses the **upper** gamma tail, and that beta is a scale,
not a rate applied directly to X. See the [SciPy parameter convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html).

For sorted observations write $u_i=F_X(x_{(i)})$. Under this fixed continuous null,
the unordered transformed observations are iid Uniform(0,1). Therefore EDF and
spacing null laws are independent of alpha and beta. With fixed equiprobable
cells the frequency vector is multinomial with probabilities $1/k$.
This conclusion does not extend automatically to arbitrary adaptive binning.
All 19 classes explicitly require both parameters; no fitting is performed.

## Formula checks

Every individual page gives the explicit implemented formula and a runnable example.

| Classes / criteria | Verified convention | Scientific source |
|---|---|---|
| KS, KS D+, KS D− | Rank discrepancies $i/n-u_i$ and $u_i-(i-1)/n$; maximum without squaring | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256); [Zhang (2002), §2.1](https://doi.org/10.1111/1467-9868.00337) |
| Anderson–Darling | Weight $(2i-1)/n$, reversed survival tail, subtraction of n | [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437); Zhang §2.2 |
| Cramér–von Mises | Half-rank target and $1/(12n)$ constant | Zhang §2.3 |
| Watson | Centered EDF integral; equivalently $W^2-n(\bar u-1/2)^2$ | [Lemeshko et al. (2014), eq. (6)](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf) |
| Kuiper, scaled Kuiper, corrected Kuiper | $V=D^++D^-$; $\sqrt n V$; $(\sqrt n+0.155+0.24/\sqrt n)V$ | Lemeshko et al., eqs. (1)–(3) |
| Liao–Shimokawa | Local maximum discrepancy divided by $\sqrt{u_i(1-u_i)}$, summed and divided by $\sqrt n$ | [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965); [Al-Zahrani (2012), eq. (13)](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf) |
| Zhang K, A, C | Half-rank likelihood terms and quarter-rank log-odds adjustment; no additional factor two | [Zhang (2002), eqs. (3.1)–(3.3), full text](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf) |
| Greenwood | Raw sum of all n+1 squared spacings, including endpoints | [Does, Helmers and Klaassen (1988), eqs. (1)–(2)](https://ir.cwi.nl/pub/1694/1694D.pdf) |
| Pearson | Power divergence at $\lambda=1$ | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| Freeman–Tukey | $\lambda=-1/2$, including factor four | Cressie and Read (1984) |
| Modified log-likelihood | $\lambda=-1$, expected-count weighting | Cressie and Read (1984) |
| Neyman modified chi-squared | $\lambda=-2$, observed-count denominator | Cressie and Read (1984) |
| Cressie–Read | $\lambda=2/3$, multiplier $9/5$ | Cressie and Read (1984) |

The frequency formulas follow by substitution or taking the limit in
$2\sum_j O_j[(O_j/E_j)^\lambda-1]/[\lambda(\lambda+1)]$, using
$\sum O_j=\sum E_j$. The implementation's special handling of zero counts
matches these limits: Freeman–Tukey contributes $4E_j$, Pearson contributes
$E_j$, and Cressie–Read contributes zero. Modified log-likelihood and Neyman
return positive infinity if any $O_j=0$ with $E_j>0$.

For Greenwood, the n+1 uniform spacings have a Dirichlet(1,…,1) law.
Each squared spacing has mean $2/[(n+1)(n+2)]$, yielding $E[G]=2/(n+2)$.
Cauchy–Schwarz and nonnegative spacings give $1/(n+1)\le G\le1$.

## Source qualifications and interpretation

The Zhang, Greenwood and Lemeshko full texts were available for formula checks.
Lemeshko's DOI resolves to **Solving Problems of Using Some Nonparametric
Goodness-of-Fit Tests**; the page references use that title.
For Liao–Shimokawa, the original bibliographic record was checked and its formula
was inspected in Al-Zahrani's published application, eq. (13); the original
publisher full text was unavailable. Cressie–Read's accessible publisher summary
and indexed paper give the divergence definition and parameter mapping; the
zero-count limits were also derived directly and checked numerically.

These are general discrepancies applied to the fixed inverse-gamma CDF, not
claims of specialized fitted inverse-gamma tests. In particular, the Weibull or
extreme-value calibration in the original Liao–Shimokawa work is inapplicable.
Kuiper corrections change normalization; they do not
produce exact finite-sample p-values. All current classes reject for large values,
including directional KS and Greenwood; this does not imply equal power against
all alternatives. Singletons and ties have well-defined values, but rounded-data
calibration differs from the continuous null.

## Numerical verification

The regression suites are:

- `tests/distributions/test_inverse_gamma.py`;
- `tests/distributions/test_inverse_gamma_correctness.py`;
- `tests/distributions/test_inverse_gamma_mathematical_validation.py`.

Independent checks cover:

- All 14 EDF/spacing classes against scalar formulas evaluated with 70-digit
  Decimal arithmetic, with four shape/scale pairs, singletons, ties and asymmetric
  probability grids. Data are constructed through reciprocal Gamma quantiles,
  independently of the production inverse-gamma CDF.
- AD, CvM and Watson against piecewise numerical integration of the empirical
  distribution discrepancy, including tied observations.
- All five frequency classes against explicit counts, varying parameters and
  empty cells, plus SciPy power-divergence comparisons with nonzero counts.
- Exact shape-one CDF $F(x)=e^{-\beta/x}$ for all five logarithmic EDF classes at
  $x=10^{20}$, where the ordinary CDF rounds to one.
- The Greenwood mean for n=1 by integration, KS directions against SciPy,
  scale equivariance, hypothesis declarations, missing parameters, positive
  support, invalid samples, input immutability and repeatability.
- A one-to-one mapping from every concrete class to a documentation page,
  scientific links, overview and navigation entries, and execution of every
  page's Python example.

Log-tail underflow is explicitly rejected without epsilon replacement. Ordinary
CDF values can still round at endpoints; finite logarithms do not guarantee a
finite final statistic if subsequent arithmetic overflows. No claim of arbitrary
precision or accuracy throughout the entire float64 input domain is made.
The frequency factory accepts numeric strings through conversion, while the
EDF/spacing sample validator rejects them.

## Reproduction and current limitations

The focused isolated run passed **707 tests**. Ruff checks passed for the new
mathematical validation suite. MkDocs built successfully; its warnings concern
other documentation families, with no inverse-gamma link warnings. The normal invocation is:

```bash
.venv/bin/python -m pytest tests/distributions/test_inverse_gamma*.py -q
```

In the current working tree, normal collection is blocked by an existing import
in `continuous/log_logistic.py` of the removed `edf.AbstractChi2Statistic`.
The isolated run bypassed only the eager `continuous/__init__.py` module; it
loaded the actual inverse-gamma implementation, factories, kernels and binning.
It does not establish that ordinary package initialization works. Reproduce it
from the repository root with:

```python
import sys
import types
from pathlib import Path
import pytest

root = Path.cwd()
name = "pysatl_criterion.statistics.goodness_of_fit.continuous"
package = types.ModuleType(name)
package.__path__ = [str(root / "src/pysatl_criterion/statistics/goodness_of_fit/continuous")]
sys.modules[name] = package
raise SystemExit(
    pytest.main(
        [
            "tests/distributions/test_inverse_gamma.py",
            "tests/distributions/test_inverse_gamma_correctness.py",
            "tests/distributions/test_inverse_gamma_mathematical_validation.py",
            "-q",
        ]
    )
)
```

The built-in Monte Carlo resolver lacks an inverse-gamma generator; external
simulation is required. Frequency configuration is included in storage identity.
Legacy directional KS storage calibration is guarded; dedicated D+ and D−
classes have distinct statistic codes. No simulated significance-level or power
study was performed in this audit.
