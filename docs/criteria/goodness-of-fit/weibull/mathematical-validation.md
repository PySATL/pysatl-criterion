# Mathematical verification of Weibull statistics

This review covers the ordinary two-parameter Weibull implementations in
`continuous/weibull.py`: 13 generated EDF classes, 5 generated frequency classes
and 15 explicit specialized classes, for a total of 33 concrete classes.
Exponentiated Weibull is a separate family.
See the [criterion catalog](index.md) for individual pages and public class names.

## Hypothesis and parameter estimation

The null model has zero location and unknown positive shape $k$ and scale $\eta$:

$$
F(x;k,\eta)=1-\exp[-(x/\eta)^k],\qquad x>0.
$$

Construct criteria with `WeibullDistributionDescriptor.DEFAULT.parse({})`.
The generated criteria fit both parameters by maximum likelihood on every call.
For ordered observations, write
$z_i=\widehat k(\log X_{(i)}-\log\widehat\eta)$ and
$u_i=1-\exp[-\exp(z_i)]$. The implementation solves the profile score in a
rescaled log domain rather than forming large powers of the original sample.
The Weibull-to-extreme-value transformation and likelihood equations are given
in [Krit (2014), §2, equations (1)–(7)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf).

Samples for generated criteria must be one-dimensional, finite, strictly
positive, nonconstant and have at least two observations. Specialized criteria
have additional sample-size, spacing or record requirements. Unknown parameters
do not imply that every specialized criterion uses MLE: some eliminate log
location and scale through correlations, normalized spacings or moments.

## Formula checks

The generated definitions below were compared with the shared kernels and
numerical references. The citations identify the discrepancy; their
fixed-parameter critical values do not automatically apply after Weibull fitting.

| Criteria | Checked definition | Scientific source |
|---|---|---|
| Kolmogorov–Smirnov D+, D−, D | $D^+=\max_i(i/n-u_i)$, $D^-=\max_i(u_i-(i-1)/n)$; $D=\max(D^+,D^-)$, without squaring or scaling | [Zhang (2002), §2.1](https://doi.org/10.1111/1467-9868.00337) |
| Anderson–Darling | $-n-n^{-1}\sum_i(2i-1)[\log u_i+\log(1-u_{n+1-i})]$; no finite-sample multiplier | [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437); fitted Weibull context: [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965) |
| Cramér–von Mises | $W^2=1/(12n)+\sum_i[u_i-(i-1/2)/n]^2$ | [Zhang (2002), §2.3](https://doi.org/10.1111/1467-9868.00337) |
| Watson | $W^2-n(\bar u-1/2)^2$, evaluated using centered deviations | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109) |
| Kuiper, scaled Kuiper, corrected Kuiper | $V=D^++D^-$; $\sqrt n V$; $(\sqrt n+0.155+0.24/\sqrt n)V$ | [Lemeshko et al. (2014)](https://doi.org/10.3103/S875669901401004X) |
| Zhang K, A, C | Likelihood-ratio discrepancies with half-ranks; Z_C uses the quarter-rank log-odds correction | [Zhang (2002), equations (3.1)–(3.3)](https://doi.org/10.1111/1467-9868.00337) |
| Liao–Shimokawa | $n^{-1/2}\sum_i\max(i/n-u_i,u_i-(i-1)/n)/\sqrt{u_i(1-u_i)}$, using MLEs | [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965) |
| Pearson, Freeman–Tukey, modified log-likelihood, Neyman chi-squared, Cressie–Read | Power divergences at $\lambda=1,-1/2,-1,-2,2/3$, respectively, using fitted bin probabilities | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |

For frequency criteria, $O_j$ is the observed bin count and $E_j=np_j>0$.
The five kernels return, respectively,

$$
\sum_j\frac{(O_j-E_j)^2}{E_j},\quad
4\sum_j(\sqrt{O_j}-\sqrt{E_j})^2,\quad
2\sum_j E_j\log\frac{E_j}{O_j},\quad
\sum_j\frac{(O_j-E_j)^2}{O_j},\quad
\frac95\sum_j O_j[(O_j/E_j)^{2/3}-1].
$$

An empty observed bin contributes $E_j$ to Pearson, $4E_j$ to Freeman–Tukey
and zero to Cressie–Read. It makes modified log-likelihood and Neyman
chi-squared infinite. These limits are part of the kernel definitions.

### Specialized implementations

The following table records the implemented definitions and the available
source support. It does not claim that all historical names reproduce a
published test exactly. Full formulas and execution settings are on the linked
criterion pages.

| Classes / short codes | Implemented definition | Source support and qualification |
|---|---|---|
| [LT2](laplace_transform2_weibull_criterion.md), [LT3](laplace_transform3_weibull_criterion.md) | $n\sum_t w_a(t)[n^{-1}\sum_i e^{-tz_i}-\Gamma(1-t)]^2$, $w_a(t)=\exp(at-e^{at})$. LT2 uses $t=-1,-1+1/m,\ldots,-1/m$; LT3 uses integer multiples of $1/m$ in $[-2.5,0.49]$ | [Krit (2014), §4, equations (22)–(23)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf); MLE variant, with no extra grid-width factor |
| [Cabana–Quiroz / CQ*](cabana_quiroz_weibull_criterion.md) | $v^T A^{-1}v$, where $v_s=\sqrt n[n^{-1}\sum_i e^{-sz_i}-\Gamma(1-s)]$, $s=(-0.1,0.02)$ and $A=((1.59,0.91),(0.91,0.53))$ | [Krit (2014), §5](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf); fixed weighting matrix, not an estimated covariance matrix |
| [Mahdi–Doostparast / MD](mahdi_doostparast_weibull_criterion.md) | Record-likelihood fitting followed by $n\int(\widehat S-S_{\mathrm{fit}})^2/F_{\mathrm{fit}}\,dF_{\mathrm{fit}}$, including both end intervals | [Doostparast (2011), equations (9), (10), (13), (14), (17)](https://arxiv.org/pdf/1110.5509); lower-record sampling, not an ordinary sorted-sample EDF |
| [Shapiro–Brain (SB)](shapiro_brain_weibull_criterion.md) | Centered ordered-log ratio $nb^2/\sum_i y_i^2$; no MLE fit; both rejection tails | [Shapiro and Brain (1987)](https://doi.org/10.1080/03610918708812586); formula checked through [Shell (2003), equation (9)](https://ipnpr.jpl.nasa.gov/progress_report/42-155/155B.pdf), with the explicit centering adjustment documented below |
| [Ozturk–Korukoglu (OK)](ozturk_korukoglu_weibull_criterion.md) | Standardized $OK_n^*$ scale ratio; no MLE; both tails | [Öztürk and Korukoğlu (1988)](https://doi.org/10.1080/03610918808812730); formula checked against [Krit (2014), §3.2.2, (3.16)–(3.20)](https://theses.hal.science/tel-01126901/document), with centered logs as described below |
| [Smith–Bain / RSB](rsb_weibull_criterion.md) | $n(1-R^2)$ with $R^2=\operatorname{corr}^2(\log X_{(i)},\log[-\log(1-i/(n+1))])$; right tail | [Krit (2014), §3.2.1, (3.12)](https://theses.hal.science/tel-01126901/document), attributed to [Smith and Bain (1976)](https://doi.org/10.1080/03610927608827337); original article not checked directly; rejection wording corrected as explained below |
| [Evans–Johnson–Green / REJG](rejg_weibull_criterion.md) | $R^2=\operatorname{corr}^2(\log X_{(i)},\log[-\log(1-p_i)])$, $p_i=(i-0.3175)/(n+0.365)$; left tail | Exact formula checked through [Krit (2014), (3.13)](https://theses.hal.science/tel-01126901/document), attributed to [Evans, Johnson and Green (1989)](https://www.govinfo.gov/content/pkg/GOVPUB-A13-PURL-gpo123139/pdf/GOVPUB-A13-PURL-gpo123139.pdf); fitted shape cancels; see qualifications below |
| [Coles / SPP](spp_weibull_criterion.md) | $\max_i\lvert h((i-1/2)/n)-h(G((y_i-\widehat\mu_B)/\widehat\sigma_B))\rvert$, $h(p)=2\arcsin\sqrt p/\pi$; Blom estimates, right tail | [Coles (1989)](https://doi.org/10.1093/biomet/76.3.593); distance checked through [Krit (2014), (3.14)–(3.15)](https://theses.hal.science/tel-01126901/document); Blom weights and approximation qualified below |
| [ST1](st1_weibull_criterion.md), [ST2](st2_weibull_criterion.md) | Rounded squared smooth components of standardized negative-log moments | [Best–Rayner–Thas (2007)](https://doi.org/10.1080/15598608.2007.10411826); coefficients checked against [EWGoF source](https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WPP.test.R) and Krit (2014), §3.2.5; see verification limits below |
| [Tiku–Singh / TS](tiku_singh_weibull_criterion.md) | $2\sum_{i=1}^{n-2}(n-1-i)g_i/(n-2)$ with approximate expected log-order spacings; both tails | [Tiku and Singh (1981)](https://doi.org/10.1080/03610928108828082); functional checked through [Sürücü (2008), (3.4)–(3.5)](https://doi.org/10.1016/j.camwa.2008.03.010); see indexing and approximation qualifications below |
| [Mann–Scheuer–Fertig / MSF](msf_weibull_criterion.md) | Upper normalized log-spacing fraction $\sum_{i=\lfloor n/2\rfloor+1}^{n-1}h_i/\sum_{i=1}^{n-1}h_i$; right tail | [Krit (2014), (29)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf), [Sürücü (2008), (4.13)](https://doi.org/10.1016/j.camwa.2008.03.010); explicit expected-order approximation, original 1973 full text not checked directly |
| [Lockhart–O’Reilly–Stephens / LOS](los_weibull_criterion.md) | Anderson–Darling functional on $q=n-2$ cumulative normalized log spacings; right tail | [Lockhart et al. (1986, JRSS B), (1)–(3)](https://doi.org/10.1111/j.2517-6161.1986.tb01418.x), checked directly in the author PDF; expected-order approximation explicitly retained |
| [Kullback–Leibler / KL](kullback_leibler_weibull_criterion.md) | Vasicek entropy estimate $-H_{m,n}-\bar z+\overline{e^z}$ on MLE-standardized logs; right tail | [Pérez-Rodríguez et al. (2009)](https://doi.org/10.1080/03610920802316658); formula checked in EWGoF 2.2.2 `WEDF.test` and `MLEst`; local default window differs; see below |

## Scope and qualifications

This document separates checking an implemented formula from establishing
its historical attribution or its null distribution. It does not establish
finite-sample rejection probabilities, nominal significance levels or power.
The Shapiro–Brain numerical formula is unchanged; its rejection region was
corrected from the left tail to both tails during the follow-up source review.

Anderson–Darling, Liao–Shimokawa and the Zhang kernels consume log-CDF and
log-survival values. This preserves tail information when ordinary CDF values
round to zero or one; representable input does not guarantee a finite statistic.
LT2/LT3 also use log-domain arithmetic to avoid overflow followed by zero-weight
multiplication. The Kuiper correction is the implemented conventional scaling,
not a proof of a fitted-Weibull finite-sample p-value approximation.

All generated criteria reject for large values. Specialized exceptions include
REJG (left tail), and Shapiro–Brain, OK and TS (both tails).

For generated EDF criteria, Monte Carlo calibration can use shape = scale = 1
because of log-location-scale invariance, but every replicate must refit both
parameters. For generated frequency criteria, the built-in resolver permits
`EqualProbabilityBins` and rebuilds the bins after each fit. Other binning rules
require external calibration with explicit simulation parameters. No automatic
chi-squared law with a parameter-count degrees-of-freedom adjustment is claimed.

MD requires acquisition order or precompressed lower records with counts.
Calibration must reproduce record extraction and the requirement of at least
two records; the generic unconditional Monte Carlo path is blocked. For other
specialized criteria, nondefault execution settings such as `m` and `a` must
also be preserved during external simulation. Unversioned stored Weibull
critical values are rejected by the current storage-calibration guards.

## Numerical verification

On 2026-10-14, `tests/statistics/test_weibull_dynamic.py` passed **247 tests**
with four warnings in an isolated package-loading run. It covers:

- Registry and pickle identity, per-criterion hypotheses, and completeness of
  all 13 EDF and 5 frequency exports without explicit class duplicates.
- Independent SciPy Gumbel fits to log observations and reference KS, AD,
  Cramér–von Mises and Liao–Shimokawa calculations.
- Shared-kernel comparisons after an independent fit. These verify integration
  with the family; they are not independent proofs of the shared formulas.
- Fitted frequency counts and expected probabilities for explicit and
  equal-probability bins, including a SciPy Pearson reference.
- Scale and positive-power transformations, repeated fitting without retained
  estimates, sample immutability, invalid input, and extreme log probabilities.
- Monte Carlo refitting, unsupported calibration modes and public stub exports.

The existing `tests/distributions/test_weibull.py` also contains references for
probability-plot correlations, spacing quadrature, moment-covariance integration,
Laplace grids, CQ*, entropy windows and record likelihood/integration. Its ordinary
run **failed during collection** on the pre-existing `AbstractChi2Statistic`
import in `continuous/beta/beta_statistics.py`; those specialized tests are not reported as
passing in this review. The isolated 247-test run bypasses
`continuous/__init__.py` and does not establish that normal package initialization
works. The test count is a dated execution result, not a claim that every
specialized formula has been independently validated.

### Shapiro–Brain follow-up verification

The formula was checked against Shell (2003), equation (9).
The implementation retains an explicit adjustment: centering removes the residual coefficient sum caused by
rounding, since $0.6079\cdot0.4228-0.2570=0.00002012$. Our algebra proves
invariance under $X\mapsto cX^d$ for positive $c,d$. No MLE is used.

Dedicated checks in `tests/statistics/test_shapiro_brain_weibull.py` compare
the statistic to an independent scalar evaluation, reproduce the published
example before centering, check the centered value separately, and test
permutation/scale/power invariance, tied samples, the degenerate $n=2$ case,
input validation and the two-tail declaration. These checks establish the
implemented calculation, not a finite-sample significance level or power claim.

The follow-up isolated run passed **268 tests**: the 247 generated-family
checks plus 21 dedicated Shapiro–Brain checks. The package-initialization
limitation described above still applies.


### Öztürk–Korukoğlu follow-up verification

The weights, ratio $b/S$ and normalization
$(b/S-1-0.13/\sqrt n+1.18/n)/(0.49/\sqrt n-0.36/n)$ agree with
[Krit (2014), equations (3.16), (3.18)–(3.20)](https://theses.hal.science/tel-01126901/document),
which attributes the test to
[Öztürk and Korukoğlu (1988)](https://doi.org/10.1080/03610918808812730).
The library centers log observations before evaluating the rounded weights:
$b=b_{\mathrm{raw}}-0.00002012\,\overline{\log X}$, as for Shapiro–Brain.
This adjustment leaves $S$ unchanged and makes both estimates scale by $d$
under $X\mapsto cX^d$, so their ratio is invariant.

Our algebra also gives
$S=\sum_{i<j}(y_j-y_i)/[n(n-1)\log 2]>0$ for nonconstant samples.
The normalization denominator is $(0.49\sqrt n-0.36)/n>0$ for $n\ge2$.
At $n=2$ the ratio is constant and provides no discrimination.
Both tails require calibration for the actual sample size; standardization
alone does not establish a standard-normal null law.

Dedicated tests in `tests/statistics/test_ozturk_korukoglu_weibull.py`
use an independent scalar evaluation of the published uncentered numerator,
apply the centering correction explicitly, and compute $S$ from pairwise
log differences. They check scale/power/permutation invariance, ties, input
immutability, invalid inputs, the two-observation degeneracy and both tails.

The isolated follow-up run passed **288 tests**: 247 generated-family,
21 Shapiro–Brain and 20 Öztürk–Korukoğlu checks. The package-initialization
limitation described above still applies. Ruff and the documentation build
also passed; the latter retains existing broken-link warnings on other pages.


### Smith–Bain follow-up verification

`SmithBainWeibullGofStatistic` agrees with the mean ranks and squared centered
correlation in [Krit (2014), §3.2.1, equation (3.12), p. 47](https://theses.hal.science/tel-01126901/document).
Krit attributes this definition to [Smith and Bain (1976)](https://doi.org/10.1080/03610927608827337).
The original article was not available for direct full-text verification.
The implementation is unchanged by this review.

Our algebra establishes the following properties. Cauchy–Schwarz gives
$0\le R^2\le1$, hence $0\le T=n(1-R^2)\le n$. The ratio $T/n$ equals the
unexplained fraction of variation in an ordinary least-squares regression with
an intercept. A right-tail rejection region is therefore appropriate.
Krit's phrase “too close to 1” following (3.12) is inconsistent with this
transformation and does not define the implemented rejection rule.

For $X=\eta E^{1/k}$, $E\sim\operatorname{Exp}(1)$, the centered log observations
are scaled by $1/k$, which cancels from the correlation. This proves that the
null law depends on $n$, but not on the two unknown Weibull parameters.
It also proves invariance under $X\mapsto aX^b$ for $a,b>0$. No MLE is needed.
The transformation is given in [Krit (2014), §2, equation (3)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf).

The scores are quantiles at mean uniform ranks, not expectations of extreme-value
order statistics. Calibration must preserve these scores, sample size and the
complete-sample observation scheme. For $n=2$, $R^2=1$ for every nonconstant
sample, so there is no discriminatory power. Ties are allowed in the calculation,
but rounding/censoring cannot be ignored when choosing a null distribution.
No universal chi-square limit or finite-sample significance level is established.

On 2026-10-14, the 20 dedicated tests in
`tests/statistics/test_smith_bain_weibull.py` passed with isolated loading of
`continuous.weibull` (bypassing `continuous/__init__.py`, as in the earlier checks).
The reference computes least-squares residuals using SciPy's minimum-Gumbel
quantiles and NumPy's linear solver, independently of the correlation helper.
Checks cover scale/power/permutation invariance, sample immutability, ties,
extreme magnitudes, exact probability-plot samples, the two-observation
degeneracy, input validation and the right-tail declaration.
The ordinary pytest run still fails during collection because `continuous/beta/beta_statistics.py`
imports the missing `AbstractChi2Statistic` from `edf`. Thus these results
validate the statistic in isolation, not normal package initialization.
Ruff passed and MkDocs built successfully, with existing warnings elsewhere.


### Evans–Johnson–Green follow-up verification

The squared correlation and ranks in `EvansJohnsonGreenWeibullGofStatistic`
agree with [Krit (2014), §3.2.1, equation (3.13), p. 47](https://theses.hal.science/tel-01126901/document).
Krit attributes it to [Evans, Johnson and Green (1989)](https://www.govinfo.gov/content/pkg/GOVPUB-A13-PURL-gpo123139/pdf/GOVPUB-A13-PURL-gpo123139.pdf).
Their report distinguishes original-scale and log-scale correlations on p. 3;
this implementation uses log observations. Its formula images were not directly
readable in this review, so the exact rank convention was checked through Krit.
The probability-plot background is [Filliben (1975)](https://doi.org/10.1080/00401706.1975.10489279);
normality tables from that article are not applicable here.

In Krit's formula, $M_i=c_i/\widehat{k}$, where
$c_i=\log[-\log(1-(i-0.3175)/(n+0.365))]$. Our algebra shows that the factor
$1/\widehat{k}$ cancels, while centering $M_i$ in the numerator changes nothing
because $\sum_i(y_i-\bar y)=0$. Thus no fit is required. The all-index rank
formula includes both endpoints without special replacements.

Cauchy–Schwarz gives $0\le R^2\le1$. Since ordered logs and scores are positively
correlated, $R^2$ measures the explained fraction of variation in linear
regression with an intercept. Small values indicate departure: the existing
left-tail declaration is correct. The result is $R^2$, not $R^4$ or
$n(1-R^2)$, and the computation is unchanged by this review.

Writing $X=\eta E^{1/k}$ with $E\sim\operatorname{Exp}(1)$ proves that the
unknown parameters cancel from the correlation. Calibration may therefore use
shape = scale = 1, with the same sample size and plotting positions, without
refitting. At $n=2$ all nonconstant samples give $R^2=1$, so there is no
power to discriminate. Tied observations are accepted, but rounded-data
calibration must model rounding. Only complete zero-location Weibull samples
are covered. This algebra does not establish a chi-square approximation or a
finite-sample significance level without calibration.

On 2026-10-14, all **24** dedicated checks in
`tests/statistics/test_evans_johnson_green_weibull.py` passed with isolated
loading of `continuous.weibull`, bypassing `continuous/__init__.py`.
The independent reference uses minimum-Gumbel quantiles and least-squares
residuals rather than the implementation's correlation helper. Tests cover
scale/power/permutation invariance, input immutability and validation, ties,
extreme magnitudes, exact probability-plot samples, the $n=2$ degeneracy,
left-tail declaration, and cancellation of the scale in Krit's uncentered
numerator. Regression checks distinguish $R^2$ from $R^4$ and the Smith–Bain
rank convention.

Ordinary pytest collection still fails because `continuous/beta/beta_statistics.py` imports
the missing `AbstractChi2Statistic` from `edf`; these results do not validate
normal package initialization. Ruff passed. MkDocs built successfully, with
existing warnings elsewhere and no warnings for the updated criterion page.


### Coles follow-up verification

The implementation now replaces MLE by Blom linear estimates. The maximum
stabilized distance matches [Krit (2014), (3.14)–(3.15), p. 48](https://theses.hal.science/tel-01126901/document).
[Coles (1989)](https://doi.org/10.1093/biomet/76.3.593) explicitly studies Blom
estimation for this distance; its separate correlation statistic is not used.
[Michael (1983)](https://doi.org/10.1093/biomet/70.1.11) introduced the stabilized
probability plot. The exact Blom linear-weight construction is checked against
[`EWGoF::BLOMEst`](https://github.com/cran/EWGoF/blob/master/R/BLOMEst.R),
not its MLE-based `WPP.test` branch. The original Coles full text was unavailable;
this is not a claim of exact agreement with its published tables.

The [criterion page](spp_weibull_criterion.md) specifies the least-squares
representation of the Blom weights and the fourth-order `GoFNS` expansion.
Its $1/3$ and $1/8$ coefficients use real division, whereas the EWGoF C++
source uses integer division. This explicit approximation convention is part
of the definition. Expected order statistics are not computed by exact quadrature.
The two-column least-squares solve avoids explicit matrix inversion, and
normalizing the input logs avoids unnecessary cancellation.

Our algebra proves equivariance of the estimates under positive affine
transformations of logs, hence invariance of the statistic to Weibull shape
and scale. At fixed $n$, simulation may use shape = scale = 1, but must
repeat the Blom fit. Since $h(p)\in[0,1]$, the statistic lies in $[0,1]$;
large values reject. No $\sqrt n$ factor, standard-normal or chi-square
calibration is asserted. At $n=2$ it is constant for nonconstant samples.
Old MLE calibration is invalid for the revised statistic; unversioned stored
critical values remain blocked by the existing guard. Tables based on other
estimators, including [Kimber (1985)](https://doi.org/10.1093/biomet/72.3.661),
are not used as calibration for this implementation.

The isolated follow-up run on 2026-10-14 passed **269 tests**: 22 dedicated
Coles checks plus 247 generated-family checks (four existing SciPy warnings).
The reference in `tests/statistics/test_coles_weibull.py` uses explicit Blom
linear weights, a scalar 2-by-2 inverse and independently written analytic
quantile derivatives, rather than the implementation's least-squares solve
and recursive derivative expressions. It checks the distance, ties, extreme
log ranges, scale/power/permutation invariance, input immutability, invalid
samples, the $n=2$ degeneracy and the right-tail declaration. A regression
check distinguishes Blom from MLE, forbids calls to MLE helpers, verifies
re-estimation on repeated calls, and checks that old stored calibration is blocked.

The previous MLE-specific SPP assertion in `tests/distributions/test_weibull.py`
has been replaced by these dedicated Blom checks. That file still fails to
collect even with isolated continuous-package loading, because
`exponentiated_weibull.py` imports the missing `AbstractChi2Statistic` from
`edf`; it is not counted among passing tests. Ordinary package initialization
also retains the previously documented `beta.py` import failure. Ruff passed
and MkDocs built successfully, with existing warnings elsewhere.


### ST1 and ST2 follow-up verification

The descriptive names are `BestRaynerThasSkewnessWeibullGofStatistic` and
`BestRaynerThasKurtosisWeibullGofStatistic`; ST1/ST2 remain aliases to the same
classes with unchanged short codes. EWGoF's source and help page disagree on
the ordering. The implementations follow its source: ST1 is the skewness
component; ST2 is the kurtosis component with a skewness correction.

The negative-log transform and divisor-n sample moments match the moment
conventions in [Best, Rayner and Thas (2007)](https://doi.org/10.1080/15598608.2007.10411826).
The numerical formulas are checked against
[EWGoF's source](https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WPP.test.R)
and [Krit (2014), §3.2.5](https://theses.hal.science/tel-01126901/document).
The accessible author PDF contains only printed pages 89–91 and 96–97, so it
does not support a direct check of the missing formula pages or appendices.

Independent scalar moment calculations on the 65 Port Pirie observations in
Table 1 give skewness 0.7110707539472332 and ST1 0.5966736536522815.
ST2 on the same data is 0.37580214526699485.
The section 4 values 0.71 for V3 squared and 0.36 for V4 squared are not
reproduced by the rounded EWGoF formulas; both numerical results are tested.
It is numerically close to the sample skewness, but this is insufficient to
conclude that the article has a typo. The formula is not adjusted to fit that
single reported number. `test_port_pirie_source_formula` locks the actual
EWGoF formula and distinguishes it from the raw skewness. The same formula
applied to the negated glass-fibre observations in Table 2 gives ST1
0.26166084645904814, rather than section 4's 0.19; full source verification
is still necessary before attributing either discrepancy to the article.

The implementation uses rounded published coefficients, rather than the
previous local delta-method normalization. The algebra of standardized moments
proves invariance under positive scaling and powers of the Weibull observations.
Both tests use right-tail rejection and finite-sample simulation. Old local
calibration is invalid and the existing storage guard remains in force.

Dedicated tests use scalar central-moment calculations, checks of the log sign,
scale/power/permutation invariance, input validation, alias identity, the n=2
degeneracy and rejection of old calibration. Normal pytest collection is blocked
by the existing `beta.py` import of missing `edf.AbstractChi2Statistic`;
isolated-module results do not validate normal package initialization.

The follow-up isolated run passed **278 tests** (31 dedicated component checks
and 247 generated-family checks; four existing SciPy warnings). Ruff and the
stub consistency check passed. MkDocs built successfully, with existing broken
links elsewhere in the documentation.


### Tiku–Singh follow-up verification

The implementation's weights and denominator agree with the generalized-spacing
$Z^*$ functional in [Sürücü (2008), (3.4)–(3.5)](https://doi.org/10.1016/j.camwa.2008.03.010).
The original sources are [Tiku (1980)](https://doi.org/10.1111/j.1467-842X.1980.tb01175.x)
and [Tiku and Singh (1981)](https://doi.org/10.1080/03610928108828082).
Their publisher records were verified; the original full texts were unavailable
for direct formula comparison.

[Krit (2014), (30)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf)
also gives TS, but uses $E_2,\ldots,E_n$ after defining intervals from $i$ to
$i+1$. Reading the formula with $E_i=h_{i-1}$ resolves this indexing mismatch.
The code includes all $n-1$ intervals in the denominator; the final interval
has zero numerator weight. No change to the executable formula was needed.

Our algebra gives $T=2\sum_{j=1}^{n-2}z_j/(n-2)$ with
$z_j=\sum_{i=1}^j g_i$. Nonnegative normalized spacings imply $0\le T\le2$;
positive spacings imply strict inequalities. At $n=3$, $T=2g_1$ is still
nondegenerate. Zero individual gaps are permitted, including the boundary
cases $T=0$ and $T=2$; a constant sample is invalid. Both tails are used.
Under $X\mapsto aX^b$, $h_i\mapsto bh_i$, so the statistic is invariant and
its null law depends on $n$, not the Weibull shape or scale. No MLE is involved.

The [criterion page](tiku_singh_weibull_criterion.md#expected-order-approximation)
records the exact beta integral and the actual `GoFNS` approximation.
The third- and fourth-central-moment coefficients are approximated as well
as truncating the Taylor series. Thus this is an approximation to the
expected-order version of the test, not an exact evaluation of its moments.
Use fresh two-tail Monte Carlo calibration of this implementation. Neither
symmetry around 1 nor the uniform-mean law for ordinary exponential spacings
is assumed for generalized log-Weibull spacings.

The follow-up run on 2026-10-14 passed **14 dedicated tests** in
`tests/statistics/test_tiku_singh_weibull.py`: an independent analytic-derivative
reference combined with the cumulative-spacing identity at four sample sizes,
shape/scale invariance, input preservation, both rejection tails, tied boundary
samples and invalid inputs. The run isolated the `continuous` package namespace
because its normal initializer fails in `beta.py` while importing the missing
`AbstractChi2Statistic` from `edf`. These results validate the TS module but
not normal package initialization. Ruff passed for the new tests; MkDocs built
successfully with existing warnings about missing documents elsewhere.


### Mann–Scheuer–Fertig follow-up verification

`MannScheuerFertigWeibullGofStatistic` replaces the abbreviated class name
without a compatibility alias, including exports and type stubs. The `MSF`
short code and executable statistic are unchanged.

The ratio agrees with [Krit (2014), equation (29)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf)
and [Sürücü (2008), equation (4.13)](https://doi.org/10.1016/j.camwa.2008.03.010).
Both attribute it to [Mann, Scheuer and Fertig (1973)](https://doi.org/10.1080/03610927308827085).
The original full text was unavailable for direct verification. The
[criterion page](msf_weibull_criterion.md) specifies the approximation used
for expected log-order spacings and distinguishes it from exact beta integration.

Our indexing check gives $r=\lfloor n/2\rfloor$ lower intervals and
$n-1-r$ upper intervals. The first upper interval is $h_{r+1}$, corresponding
to zero-based `g[r]`; the final interval is included. In particular, even $n$
has one fewer upper interval. There is no off-by-one correction to make.
Nonnegative gaps give $0\le S\le1$, with strict bounds for distinct samples;
$n=3$ remains nondegenerate. An equivalent identity is $S=1-z_r$,
where $z_r=\sum_{i=1}^r h_i/\sum_{i=1}^{n-1}h_i$.

Under $X\mapsto aX^b$, $h_i\mapsto bh_i$, proving invariance to the unknown
Weibull parameters. The right-tail rule is an explicit one-sided choice,
with possible poor power against alternatives producing small ratios.
It must not be described as a two-tail omnibus test. Calibrate at fixed $n$
by simulation at shape = scale = 1, repeating the same approximation.
Expected spacings alone do not establish independence or an exact beta law;
no exact beta/F distribution or chi-square limit is used. No censoring API
or agreement with historical critical-value tables is claimed.

The follow-up run on 2026-10-14 passed **280 isolated tests**: 19 dedicated
MSF checks, 14 TS checks and 247 generated-family checks (four existing SciPy
warnings). `tests/statistics/test_mann_scheuer_fertig_weibull.py` verifies
independent analytic quantile derivatives and the complementary cumulative
spacing identity, even/odd splits, endpoint samples, invariance, input
preservation, rejection direction, the public name and invalid inputs.
The run bypassed `continuous/__init__.py`, whose normal import still fails
in `beta.py` on the missing `AbstractChi2Statistic` import from `edf`.
It validates the criterion module, not normal package initialization.
Ruff passed for the new tests; MkDocs built with existing missing-document
warnings elsewhere and none for the MSF page.


### Lockhart–O’Reilly–Stephens follow-up verification

The class is now `LockhartOReillyStephensWeibullGofStatistic` in implementation,
exports and type stubs, without an alias for the abbreviated name. The `LOS`
short code and executable formula are unchanged.

We checked equations (1)–(3), pp. 344–345 directly in the
[author copy of Lockhart, O’Reilly and Stephens (1986, JRSS B)](https://www.sfu.ca/~lockhart/Research/Papers/LockhartOReillyStephensJRSSB86.pdf).
For a complete sample, their $k=1$ and $r=n-2$ give $n-1$ normalized gaps,
$n-2$ interior cumulative values, and precisely the implemented AD sum.
There is no off-by-one error or missing finite-sample correction.
The [companion paper](https://doi.org/10.1002/nav.3800330307) provides Weibull
critical-value and power studies; its full text was not checked directly.

The [criterion page](los_weibull_criterion.md) records the exact beta integral
and the explicit fourth-derivative `GoFNS` approximation. Higher beta central
moments are also approximated; this does not reproduce exact expected order
statistics or establish agreement with historical tables.

Our integral identity $A_S^2=q\int_0^1[H_q(u)-u]^2/[u(1-u)]\,du$ verifies
nonnegativity and the logarithmic boundary divergence. Ties at either extreme
give infinity, while interior ties can remain finite. At $n=3$, the formula
reduces to $-1-\log[z_1(1-z_1)]$, so it is not degenerate. Reverse cumulative
sums retain the small upper-tail mass when direct subtraction would lose it.

Positive affine transformations of logs cancel in the normalized spacings,
so complete Weibull shape = scale = 1 samples suffice for calibration at
fixed $n$. No MLE fit is involved. The authors explicitly explain on p. 345
that general normalized spacings do not yield uniform order statistics,
even via a naive asymptotic independence argument. Thus ordinary uniform AD
tables are inappropriate. Use the right tail of simulations of this same
approximate statistic; no chi-square law or censoring calibration is asserted.

The follow-up run on 2026-10-14 passed **300 isolated tests**: 20 dedicated
LOS checks, 19 MSF checks, 14 TS checks and 247 generated-family checks
(four existing SciPy warnings). The LOS reference uses independently written
analytic quantile derivatives and numerical quadrature of the weighted EDF
integral, rather than the production logarithmic sum. Additional tests cover
endpoint/interior ties, preservation of tiny survival mass, invariance, input
preservation, invalid samples and removal of the old public name.
The runner bypassed `continuous/__init__.py` because normal initialization
fails in `beta.py` while importing the missing `AbstractChi2Statistic` from
`edf`; this does not validate normal package initialization. Ruff passed for
the new tests. MkDocs built with existing warnings elsewhere and no warning
for the LOS page.


### Kullback–Leibler follow-up verification

The statistic matches the MLE KL functional in the pinned EWGoF 2.2.2
[`WEDF.test`](https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WEDF.test.R)
for the same effective entropy window. Its scientific origin is
[Pérez-Rodríguez, Vaquera-Huerta and Villaseñor-Alva (2009)](https://doi.org/10.1080/03610920802316658),
with [Vasicek entropy estimation](https://doi.org/10.1111/j.2517-6161.1976.tb01566.x)
and [Song's KL methodology](https://doi.org/10.1109/18.995548).
The 2009 full text was not available for direct verification; executable
formula correspondence uses EWGoF rather than an assertion of identical
published tables.

Our change-of-variable calculation gives minimum Gumbel for $Y=\log X$,
$\mu=\log\eta$, $\sigma=1/k$. Therefore $\log g(z)=z-e^z$ and the returned
$-H-\bar z+\overline{e^z}$ has the correct signs. EWGoF's
[`MLEst`](https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/MLEst.R)
starts with negative logs but reflects its standardized output again, giving
the same orientation. Reflection and endpoint repetition preserve the entropy
sum; the maximum-Gumbel formulation on negative logs is equivalent.
At the MLE, $\overline{e^z}=1$. There is no extra factor $n$, exponentiation
or nonnegativity clipping. Estimated entropy can produce negative KL estimates.

The local window is an integer $1\le m<n/2$, defaulting to
$\min(\lfloor\sqrt n\rfloor,\lfloor(n-1)/2\rfloor)$. EWGoF instead overwrites
`paramKL` with a piecewise rule ($n=20$: 3 versus our 4), with uncovered
boundary sizes falling through to 14. These conventions are not equivalent.
The local profile-score solver also differs from EWGoF's bounded optimizer.
No executable formula or window change was needed; the documented variant
now explicitly states these differences.

MLE equivariance under positive affine changes of logs proves invariance to
Weibull shape and scale at fixed $m$. Every Monte Carlo replicate must refit
both parameters and preserve the window. Custom execution windows require
matching external calibration. Zero window gaps give infinity; individual
ties with positive window gaps are allowed. The criterion is right-tailed,
with no unqualified normal/chi-square calibration or censoring support.

The follow-up run on 2026-10-14 passed **322 isolated tests**, including 22
new KL checks and 300 existing spacing/generated-family checks (four existing
SciPy warnings). The new reference fits `scipy.stats.gumbel_l` to raw logs
and evaluates the unstandardized log density with independently padded entropy
windows. A separate `gumbel_r` fit to negative logs verifies reflection;
a scalar translation of EWGoF's inner KL branch verifies matched-window
algebra, not an execution of R. Tests also check the MLE score identity,
default-window differences, invariance, input preservation, ties and invalid
inputs. The runner bypasses `continuous/__init__.py` due to the existing
`beta.py` import of the missing `AbstractChi2Statistic` from `edf`; this run
does not validate normal package initialization. Ruff passed for the new
tests. MkDocs built with existing warnings elsewhere, none for the KL page.
