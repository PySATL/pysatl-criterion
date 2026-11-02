# Mathematical verification of log-normal statistics

Audit date: 2026-10-17. Scope: **all 55 concrete classes** exported by
`statistics.goodness_of_fit.continuous.log_normal`: 15 EDF variants, five
frequency statistics, one spacing statistic, 32 logarithmic normality wrappers
and two KL functionals. The [catalog](index.md) lists every class, full code,
hypothesis, rejection tail and individual documentation page.

## Findings and corrections

1. **Shape tests were presented as tests of fixed parameter values.** Normality
   statistics on logs and both KL statistics eliminate log-location and log-scale.
   They now accept an empty hypothesis in either coordinate system; Monte Carlo
   sampling explicitly selects mu=0,s=1. Fully fixed schemas remain accepted for
   compatibility and remain in `hypothesis()`, but these shape statistics are
   invariant to those values. They cannot distinguish two log-normal distributions
   with different parameters. The 21 fixed-CDF/spacing/frequency tests continue to
   require both parameters. This is an explicit contract distinction, not implicit
   parameter filling or a claim that every shape test is omnibus.
2. **Invalid logarithmic data could become a rejection-tail error.** A universal
   `+inf` penalty is inappropriate for lower-tail tests such as Shapiro–Wilk and
   Filliben. Logarithmic wrappers and KL now raise `ValueError` for nonpositive
   input. Empty, multidimensional, nonfinite, complex and actively masked data
   are rejected before conversion. Quesenberry–Miller now validates its input too.
   EDF and frequency behavior follows their mathematical support/zero-count limits.
3. **KL powers and cancellations were unstable.** Forming X**(1/sigma_hat) could
   overflow solely because the measurement unit changed. KL now uses centered,
   standardized logs, log-sum-exp and a near-zero cumulant series. The former
   near-zero expression used uncentered log moments and the wrong coefficient;
   the squared limiting integrand is m3²/6, not its former expression. No
   integration warnings are suppressed, and zero dispersion is rejected.
4. **The general-alpha KL threshold regression could return impossible negative
   thresholds.** For example, the old supremum expression at n=50, alpha=0.025
   gave approximately -103.91 although the statistic is nonnegative. That branch
   now raises `NotImplementedError`. The historical alpha=0.01/0.05 formulas
   remain available for compatibility, explicitly labelled unverified regressions.
   Their finite-sample level and published domain are not certified by this audit.
   Nonpositive or nonfinite regression results now raise `ValueError` too.
5. **Documentation coverage was incomplete.** There were 40 pages for 55 classes,
   without a complete catalog or a log-normal validation report. The pages now
   state formulas, hypotheses, tails, input restrictions, executable examples and
   scientific references; all 15 missing pages and MkDocs entries were added.
   References to the implementation itself were replaced by scientific sources.
6. **Shared kernels and normality definitions were retained.** Direct CDF
   evaluation, KS directions, Zhang–Wu's 10*Z_A-32 normalization and frequency
   zero-count limits passed independent comparisons. Qualifications for Hosking,
   GLB and approximate normal-score methods are retained, not silently certified.

## Mathematical reduction and hypothesis conventions

For X>0, log(X) ~ N(mu,s²) if and only if X is log-normal. There is no density
Jacobian in a test statistic evaluated on transformed observations: the normality
functional is simply T(log(X_1),...,log(X_n)). For location-scale-invariant T,
T(a+b*log(X))=T(log(X)) for b>0. This proves the transfer of the implemented
functional and its null distribution, but does not prove consistency or verify
the scientific provenance of a legacy T.

For a fully specified null set u_i=Phi((log(x_(i))-mu)/s). The probability
integral transform reduces fixed-CDF EDF and spacing criteria to their uniform
functionals. No mean/variance is fitted. In shape statistics the normality
functional may use s0 (ddof=0), s1 (ddof=1), robust scales or other estimates;
those individual conventions are preserved. They must not be confused with
the specified log-normal shape parameter s. `scale=exp(mu)` is the median;
the arithmetic mean of X is exp(mu+s²/2).

All statistics reject in the tail listed in the catalog. Signed two-tail
statistics need two simulated quantiles; symmetry is not assumed. D+ and D−
are directional CDF discrepancies but both reject **large** values. Frequency
statistics use O_j and E_j=n*p_j in original sample units; only fixed positive
cell probabilities justify the ordinary multinomial k-1 asymptotic reference.
Adaptive partitions require calibration of the whole binning procedure.

## Formula-by-formula coverage

Every row below corresponds to one concrete exported class. “Local” means that
an independently checked mathematical functional is available but equality to
the original named publication or its calibration is unresolved. The independent
normality tests are rerun along with log-transform composition tests; composition
checks alone are not presented as independent verification of an underlying
normality formula. Related references are distinguished from exact provenance.

| Code and page | Definition checked | Qualification / evidence | Scientific source |
| --- | --- | --- | --- |
| [`AD`](anderson_darling_log_normal_criterion.md) | Weighted EDF integral; density and EDF quadrature using the fixed mu and s | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Anderson & Darling (1954)](https://doi.org/10.1080/01621459.1954.10501232); [Stephens (1976)](https://doi.org/10.1214/aos/1176343411) |
| [`BS`](bonett_seier_log_normal_criterion.md) | sqrt(n+2)*(13.29*log(s0/mean(abs(x-mean(x))))-3)/3.54; original eqs. (5)–(6) | Geary-kurtosis component; both tails | [Bonett & Seier (2002)](https://doi.org/10.1016/S0167-9473(02)00074-9) |
| [`BM1`](bontemps_meddahi1_log_normal_criterion.md) | Sum of squared normalized Hermite means for orders 3–4; scipy.special evaluation | s1 finite-sample convention; iid only | [Bontemps & Meddahi (2005)](https://doi.org/10.1016/j.jeconom.2004.02.014) |
| [`BM2`](bontemps_meddahi2_log_normal_criterion.md) | Same construction for orders 3–6; scipy.special evaluation | s1 finite-sample convention; iid only | [Bontemps & Meddahi (2005)](https://doi.org/10.1016/j.jeconom.2004.02.014) |
| [`CC1`](cabana_cabana1_log_normal_criterion.md) | Continuous supremum of the skewness-focused Hermite process; scalar optimization versus polynomial roots | Truncation five; no claim of consistency against every alternative | [Cabaña & Cabaña (2003)](https://doi.org/10.1023/A:1026235220018) |
| [`CC2`](cabana_cabana2_log_normal_criterion.md) | Continuous supremum of the kurtosis-focused Hermite process; scalar optimization | Truncation five | [Cabaña & Cabaña (2003)](https://doi.org/10.1023/A:1026235220018) |
| [`CS`](chen_shapiro_log_normal_criterion.md) | sqrt(n)*(1-mean(normalized adjacent spacings)/s1) | Approximate normal scores | [Chen & Shapiro (1995)](https://doi.org/10.1080/00949659508811711) |
| [`CHI2_PEARSON`](chi2_pearson_log_normal_criterion.md) | Power divergence lambda=1; independent scipy.stats.power_divergence | Fixed null; independent multinomial power divergence; zero-count limits | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`COIN`](coin_log_normal_criterion.md) | Squared cubic OLS coefficient on normal order scores; independent order-density quadrature and least squares | Approximation: nscor2, 4<=n<=2000; original finite-sample implementation not fully certified | [Coin (2008)](https://doi.org/10.1016/j.csda.2007.07.012) |
| [`CVM`](cvm_log_normal_criterion.md) | Integral of squared EDF discrepancy; independent quadrature and scipy.stats.cramervonmises | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Cramér (1928)](https://doi.org/10.1080/03461238.1928.10416872) |
| [`CRESSIE_READ`](cressie_read_log_normal_criterion.md) | Power divergence lambda=2/3, including factor 9/5 | Fixed null; independent multinomial power divergence; zero-count limits | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`DAP`](dagostino_pearson_log_normal_criterion.md) | Z_skew²+Z_kurtosis²; scipy.stats.normaltest | Upper tail, not signed skewness | [D’Agostino & Pearson (1973)](https://doi.org/10.1093/biomet/60.3.613) |
| [`D`](dagostino_log_normal_criterion.md) | Normalized rank-weighted contrast; independent pairwise absolute-distance representation | Rounded asymptotic centering/scaling; both tails | [D’Agostino (1971)](https://doi.org/10.1093/biomet/58.2.341) |
| [`DLDMZEPD`](desgagne_lafaye_log_normal_criterion.md) | n*r.T*A*r; quadrature of score means, covariance and scale-score projection | Ill-conditioned matrix and rounded coefficients; rel tolerance 2e-4 | [Desgagné, Lafaye de Micheaux & Leblanc (2013)](https://doi.org/10.1080/03610926.2011.577548) |
| [`DH`](doornik_hansen_log_normal_criterion.md) | Squared transformed skewness plus conditional kurtosis; published appendix numerical example | n>=8; finite-sample chi-square approximation is not exact | [Doornik & Hansen (2008)](https://doi.org/10.1111/j.1468-0084.2008.00537.x) |
| [`EP`](epps_pulley_log_normal_criterion.md) | Gaussian-weighted empirical-characteristic-function distance; numerical integration | z0 standardization | [Epps & Pulley (1983)](https://doi.org/10.1093/biomet/70.3.723) |
| [`FILLI`](filliben_log_normal_criterion.md) | Correlation with approximate normal order medians; scipy.stats.probplot | Approximate plotting positions | [Filliben (1975)](https://doi.org/10.1080/00401706.1975.10489279) |
| [`FREEMAN_TUKEY`](freeman_tukey_log_normal_criterion.md) | Power divergence lambda=-1/2, including factor 4 | Fixed null; independent multinomial power divergence; zero-count limits | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`GMG`](gmg_log_normal_criterion.md) | s0 divided by sqrt(pi/2)*mean(abs(x-median(x))) | Directed heavy-tail test; upper tail | [Gel, Miao & Gastwirth (2007)](https://doi.org/10.1016/j.csda.2006.08.022) |
| [`GLB`](glen_leemis_barr_log_normal_criterion.md) | Beta order-CDF transformation and reversed-weight logarithmic sum; binomial-tail identity | Local fitted-normal functional; exact author attribution unconfirmed | [Glen, Leemis & Barr (2001)](https://doi.org/10.1109/24.963129); exact attribution unconfirmed |
| [`HOSKING1`](hosking1_log_normal_criterion.md) | Quadratic ratios of L-moments with trim 0; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`HOSKING2`](hosking2_log_normal_criterion.md) | Quadratic ratios of L-moments with trim 1; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`HOSKING3`](hosking3_log_normal_criterion.md) | Quadratic ratios of L-moments with trim 2; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`HOSKING4`](hosking4_log_normal_criterion.md) | Quadratic ratios of L-moments with trim 3; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`JB`](jarque_bera_log_normal_criterion.md) | n*(skew²/6+excess_kurtosis²/24); scipy.stats.jarque_bera | Asymptotic calibration requires care at small n | [Jarque & Bera (1987)](https://doi.org/10.2307/1403192) |
| [`KL_INT`](kl_integral_log_normal_criterion.md) | n times integral of (D/sigma)^2 on [-5,5] | Truncated numerical functional; original full-text normalization and critical regressions not certified | [Batsidis, Economou and Tzavelas (2016)](https://doi.org/10.1080/00949655.2014.1003138) |
| [`KL_SUP`](kl_supremum_log_normal_criterion.md) | sqrt(n) max |D/sigma| on 1000 equally spaced r in [-5,5] | Truncated numerical functional; original full-text normalization and critical regressions not certified | [Batsidis, Economou and Tzavelas (2016)](https://doi.org/10.1080/00949655.2014.1003138) |
| [`KS_MINUS`](kolmogorov_smirnov_d_minus_log_normal_criterion.md) | max(u_i-(i-1)/n); independent density quadrature | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256); [Lilliefors (1967)](https://doi.org/10.1080/01621459.1967.10482916) (parameter-fitting context) |
| [`KS_PLUS`](kolmogorov_smirnov_d_plus_log_normal_criterion.md) | max(i/n-u_i); independent density quadrature | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256); [Lilliefors (1967)](https://doi.org/10.1080/01621459.1967.10482916) (parameter-fitting context) |
| [`KS`](kolmogorov_smirnov_log_normal_criterion.md) | max(D+,D−) against specified normal CDF; scipy.stats.kstest in all directions | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256) |
| [`KUIPER_CORRECTED`](kuiper_corrected_log_normal_criterion.md) | (sqrt(n)+0.155+0.24/sqrt(n))*(D++D−) | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60)50006-0); [Lemeshko et al. (2014)](https://doi.org/10.3103/S875669901401004X) |
| [`KUIPER`](kuiper_log_normal_criterion.md) | D++D− | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60)50006-0) |
| [`KUIPER_SCALED`](kuiper_scaled_log_normal_criterion.md) | sqrt(n)*(D++D−) | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60)50006-0); [Lemeshko et al. (2014)](https://doi.org/10.3103/S875669901401004X) |
| [`KURTOSIS`](kurtosis_log_normal_criterion.md) | Signed Anscombe–Glynn cube-root transform; scipy.stats.kurtosistest | n>=5; approximate calibration, especially below 20 | [Anscombe & Glynn (1983)](https://doi.org/10.1093/biomet/70.1.227) |
| [`LIAO_SHIMOKAWA`](liao_shimokawa_log_normal_criterion.md) | Sum of local EDF maxima weighted by 1/sqrt(u*(1-u)*n) | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Liao & Shimokawa (1999)](https://doi.org/10.1080/00949659908811965) |
| [`LG`](looney_gulledge_log_normal_criterion.md) | Correlation with Blom normal scores; independent correlation matrix | Approximate plotting positions | [Looney & Gulledge (1985)](https://doi.org/10.1080/00031305.1985.10479395) |
| [`MI`](martinez_iglewicz_log_normal_criterion.md) | Classical median-centered scale divided by truncated biweight scale; direct outlier calculation | Positive MAD and nonzero biweight denominator required | [Martinez & Iglewicz (1981)](https://doi.org/10.1093/biomet/68.1.331) |
| [`MODIFIED_LOG_LIKELIHOOD`](modified_log_likelihood_log_normal_criterion.md) | Power divergence lambda=-1; explicit empty-cell limit | Fixed null; independent multinomial power divergence; zero-count limits | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`CHI2_NEYMAN`](neyman_chi2_log_normal_criterion.md) | Power divergence lambda=-2; explicit empty-cell limit | Fixed null; independent multinomial power divergence; zero-count limits | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`QUESENBERRY_MILLER`](quesenberry_miller_log_normal_criterion.md) | Sum of squared n+1 spacings plus adjacent products, with no cyclic product | Fixed-CDF probability integral transform; hand-calculated example | [Quesenberry and Miller (1977)](https://doi.org/10.1080/00949657708810150) |
| [`RJB`](robust_jarque_bera_log_normal_criterion.md) | n*m3²/(6*J^6)+n*(m4/J^4-3)²/64; direct moments and median absolute deviations | Upper tail; J uses the median | [Gel & Gastwirth (2008)](https://doi.org/10.1016/j.econlet.2007.05.022) |
| [`RJ`](ryan_joiner_log_normal_criterion.md) | Normal-score correlation; independent plotting positions for every alpha and averaged tied ranks | Weighted/alpha options change calibration | [Ryan & Joiner (1976), technical report](https://www.additive-net.de/en/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner) |
| [`SF`](sf_log_normal_criterion.md) | Squared Blom-score correlation | Approximation to expected normal order statistics | [Shapiro & Francia (1972)](https://doi.org/10.1080/01621459.1972.10481232); [Weisberg & Bingham (1975)](https://doi.org/10.1080/00401706.1975.10489283) |
| [`SWRG`](swrg_log_normal_criterion.md) | Normalized score contrast squared; inverse Brownian-bridge quantile covariance calculation | Approximate order-score covariance; lower tail | [Rahman & Govindarajulu (1997)](https://doi.org/10.1080/02664769723828) |
| [`SW`](shapiro_wilk_log_normal_criterion.md) | Royston approximation to Shapiro–Wilk W; SciPy at n=3,4,5,6,8,20,100,1000 | Coefficient approximation, not exact covariance inversion | [Shapiro & Wilk (1965)](https://doi.org/10.1093/biomet/52.3-4.591); [Royston (1995)](https://doi.org/10.2307/2986146) |
| [`SKEW`](skew_log_normal_criterion.md) | Signed D’Agostino transform; scipy.stats.skewtest and exact zero-skew regression | n>=8; both tails | [D’Agostino (1970)](https://doi.org/10.1093/biomet/57.3.679) |
| [`SH`](spiegelhalter_log_normal_criterion.md) | Power combination of standardized range and absolute deviations; direct powers versus log-space code | Documented symmetric-alternative functional; original full text not inspected | [Spiegelhalter (1977)](https://doi.org/10.1093/biomet/64.2.415) |
| [`WATSON`](watson_log_normal_criterion.md) | Centered EDF squared integral | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109) |
| [`ZHANG_ZA`](zhang_a_log_normal_criterion.md) | Reciprocal-rank weighted logarithmic tails | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [`ZHANG_ZC`](zhang_c_log_normal_criterion.md) | Squared log-odds discrepancies with quarter-rank correction | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [`ZHANG_ZK`](zhang_k_log_normal_criterion.md) | Maximum rank-weighted binary log-likelihood discrepancy | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [`ZQ`](zhang_q_log_normal_criterion.md) | log of mean endpoint spacings divided by mean lag-four spacings | n>=8; both tails; exact original text not inspected | [Zhang (1999)](https://doi.org/10.1080/02664769922395) |
| [`ZQS`](zhang_q_star_log_normal_criterion.md) | ZQ applied to the reflected sample; independent spacing expression | n>=8; both tails | [Zhang (1999)](https://doi.org/10.1080/02664769922395) |
| [`ZWA`](zhang_wu_a_log_normal_criterion.md) | 10*Zhang_A-32; density-quadrature CDF and independent rank sum | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Zhang & Wu (2005)](https://doi.org/10.1016/j.csda.2004.05.034) |
| [`ZWC`](zhang_wu_c_log_normal_criterion.md) | Zhang_C log-odds discrepancy; independent rank sum | Fixed-CDF adaptation; independent density/EDF quadrature or rank sum | [Zhang & Wu (2005)](https://doi.org/10.1016/j.csda.2004.05.034) |

## KL derivation, numerical definition and limits of verification

Let z_i be centered log observations standardized with ddof=0, K(r) the empirical
log moment-generating function and kappa_k its cumulants. With
D(r)=2*K(r)-r*K'(r), its power series begins

$$D(r)=-\frac{\kappa_3r^3}{6}-\frac{\kappa_4r^4}{12}+O(r^5).$$

The denominator satisfies

$$\sigma^2(r)=e^{r^2}(r^4-3r^2+4)-(4+r^2)
=r^6\sum_{j=0}^{\infty}\frac{(j+1)^2r^{2j}}{(j+3)!}.$$

Thus R(r)=D(r)/sigma(r) has opposite one-sided limits when skewness is nonzero,
but its absolute value and square have continuous extensions. In particular,
R(0+)=-kappa3/sqrt(6), R(0+)²=kappa3²/6. The implementation uses these extensions,
not a denominator floor. Centering cancels algebraically: replacing log data
by log data+c adds cr to K and c to K', cancelling the explicit r*mean term in
the uncentered formula. Positive rescaling cancels through the fitted log SD.

`KL_SUP` retains the legacy 1000-point equally spaced grid on [-5,5]; this is
only a **grid approximation** to a finite-window supremum. `KL_INT` integrates
n*R(r)² on [-5,5] with adaptive quadrature, split at zero and absolute/relative
tolerance 1e-9. No error bound for omitted infinite tails or missed grid maxima
is asserted. Decimal 70-digit direct expressions independently check both signs
of r, including +/-1e-6, plus the integral; scale tests shift log data by +/-600.
The cumulant expansion is used through order 10 when |r|*max|z|<=0.05.

[Batsidis, Economou and Tzavelas (2016)](https://doi.org/10.1080/00949655.2014.1003138)
provides the KL size-weighted symmetry motivation. The publisher abstract and
author bibliography were available; the primary full text was not. Consequently
this audit verifies the **documented truncated functionals**, not the exact
paper's domain, normalization, critical-value regressions or reported power.
The old test comments calling two data values “from the article” cannot serve
as independent provenance without a pinpoint source. Those rounded regressions
are retained as compatibility checks, not used as the mathematical proof.

## Scientific sources and inherited qualifications

The accessible [Zhang (2002) paper, §§2–3](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf)
provides the fixed-null EDF expressions and likelihood-ratio ranks. The
noncyclic Quesenberry–Miller definition is available in the authors' comparative
study, [Sreedevi and Kattumannil, §4.1](https://arxiv.org/html/2106.06368v1#S4.SS1).
Original bibliographic links appear in every page. A bibliography entry or an
accessible abstract is not claimed to be a full-text formula check.

For the 32 wrappers, the detailed scientific-source audit and independent
functional constructions are in the [normality report](../normality/mathematical-validation.md).
Their numerical verification was rerun; inherited limitations remain:

- **Hosking 1–4:** L-moments agree with exhaustive order-statistic combinations.
  The three local rows of centering/variance constants have no established
  primary-source calibration here and do not shrink as an asymptotic variance
  sequence. They define local discrepancies; simulate at the actual n.
- **GLB:** the fitted beta/order-probability transformation and reversed log
  weights agree with the documented local expression. Equality to the original
  Glen–Leemis–Barr test remains unconfirmed; do not use its tables by name alone.
- **Coin, SW, SF, SWRG, correlations and spacings:** normal order scores or
  their covariance are approximated. Coin retains its 4<=n<=2000 restriction.
  These are not exact finite-sample normal covariance inversions.
- **Hermite/moment and directed shape tests:** finite-order moments and directed
  alternatives do not guarantee power against every nonnormal log distribution.
- **Zhang–Wu variants:** historical normalization is retained; the log-normal
  versions use a fixed CDF rather than fitted normality probabilities.
- **Liao–Shimokawa:** a fixed-log-normal application of the general EDF functional;
  the original extreme-value-family calibration is not transferred.

## Numerical checks and reproducibility

Result: **1576 tests passed** with ordinary imports. Ruff and stub freshness
checks passed. MkDocs built successfully; its existing broken-link warnings
concern other families, not the log-normal pages. This count includes the
normality reference suite and hypothesis-generator checks listed below.

The focused checks exercise all 55 public statistics and both coordinate systems.
They include independent normal-density and EDF quadrature, rank sums, SciPy
multinomial power divergences, hand-calculated spacing sums, 70-digit KL
expressions, normality functional references, log-affine invariance, invalid
samples, unchanged input, hypothesis preservation and composite-null sampling.

Typical checked tolerances are rel=2e-9, abs=2e-10 for EDF identities;
rel=3e-8, abs=2e-9 for log-transform composition at large log offsets;
rel=2e-7, abs=2e-9 for KL pointwise comparisons, and rel=2e-8 for KL integration.
These bound the tested cases, not a global error or exact significance level.
No Monte Carlo level/power study was performed. In particular, the legacy KL
threshold regressions and unresolved historical calibrations are not certified.

Run from the repository root using ordinary package imports (no isolation shim):

```sh
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/distributions/test_log_normal.py \
  tests/distributions/test_log_normal_parameters.py \
  tests/distributions/test_log_normal_mathematical_validation.py \
  tests/distributions/test_log_normal_documentation.py \
  tests/statistics/test_log_normal_factory.py \
  tests/distributions/test_normal_correctness.py \
  tests/distributions/test_normal_mathematical_validation.py \
  tests/utils/test_hypothesis_generator.py -q
.venv/bin/python scripts/generate_log_normal_stub.py --check
.venv/bin/python -m mkdocs build --site-dir /tmp/pysatl-log-normal-audit
```

## Documentation coverage

The catalog, report, navigation and 55 individual pages are checked against
runtime exports. Every page names its exact Python class and code, supplies an
executable example and scientific links, and links back to this report. Python
examples and public-class docstring examples execute in tests; local Markdown
links are checked. `log_normal/__init__.pyi` is regenerated and checked with `--check`.

## Numba optimization verification

Optimization checked on 2026-10-23. All 55 public criteria reach numerical
kernels compiled with `@jit(nopython=True, cache=True)`. EDF, frequency and
normality wrappers reuse their existing compiled kernels. Quesenberry–Miller
now compiles sorting and the spacing sum; KL compiles cumulant coefficients,
the stable ratio, the entire 1000-point supremum loop and the squared integrand.
SciPy retains CDF/log-tail evaluation and adaptive quadrature with the same
window, tolerance and split at zero. No fast-math reassociation is enabled.
The public call is therefore not claimed to be wholly compiled.

All implementation helpers, including validation, class generation and factory
adapters, are methods inside classes in `log_normal/log_normal_statistics.py`.
There are no module-level function definitions. The public import path and
statistic names are unchanged; the package stub describes the class methods.

The 117 tests in `tests/distributions/test_log_normal_numba.py` verify actual
nopython signatures, disk-cache configuration, compiled/Python kernel agreement,
unchanged input and execution through every public criterion on continuous and
tied samples. Together with the mathematical, documentation and generator suites
listed above, **1693 tests passed**. Ruff and stub freshness checks also passed.

A local comparison against the saved pre-optimization implementation used
n=200, seed 123 and the minimum of three batches of ten warmed public calls.
All 55 values agreed (`rtol=2e-8`, `atol=2e-9`). Measured results for the changed
numerical paths were:

| Statistic | Before, microseconds | After, microseconds | Speedup |
| --- | ---: | ---: | ---: |
| KL supremum | 42886.4 | 1491.3 | 28.76x |
| KL integral | 6543.8 | 619.8 | 10.56x |
| Quesenberry–Miller | 38.0 | 30.2 | 1.26x |

These are local warmed timings, not performance guarantees. Existing shared
kernels were already compiled. First-call timings include compilation or disk
cache loading and are reported separately by the benchmark:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_log_normal_pipeline.py --size 200 --number 10
PYTHONPATH=src .venv/bin/python -m pytest tests/distributions/test_log_normal_numba.py -q
```

The benchmark's optional `--baseline /path/to/saved/log_normal.py` compares
against a saved prior module and checks values before reporting speedups.
