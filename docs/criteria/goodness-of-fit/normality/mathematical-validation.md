# Mathematical verification of normality statistics

Audit date: 2026-10-17. Scope: all **60 concrete classes** exported by
`statistics.goodness_of_fit.continuous.normal`, including the generated EDF
and frequency classes. The [catalog](index.md) lists the exact Python names,
storage codes, hypotheses and rejection tails.

The audit checks implemented mathematical functionals, numerical behavior,
hypothesis contracts and documentation. It does **not** certify exact
finite-sample p-values, universal power or every historical author attribution.
In particular, eleven legacy classes (four Hosking, six graph and GLB) have
qualifications described below. They remain available under their existing names.

## Findings and changes

- No discrepancy was found between the implemented formulas and the independent
  numerical definitions on the tested cases. For some legacy classes the
  independently checked definition is the library's documented functional,
  rather than a fully verified reconstruction of the original publication.
- Masked observations were silently converted to ordinary arrays by most normal
  statistics, exposing the hidden values to the calculation. `_normal_sample`
  now rejects observations with an active mask, consistent with frequency tests.
- The D’Agostino–Pearson page incorrectly documented `SkewNormalityGofStatistic`,
  a signed skewness test. It now documents `DAPNormalityGofStatistic`, the sum
  of squared skewness and kurtosis transforms, and its upper rejection tail.
- Fifteen generated criteria had no page. They now have separate pages with
  formulas, examples, sources and fitted-normal calibration instructions.
- Scientific context was missing for Hosking, GLB and graph pages. References
  have been added without claiming that related theory establishes their exact
  local normality tests. Broken links to `normal-statistics-review.md` were
  replaced by links to this report and the catalog.
- Generated class docstrings now distinguish fixed and fitted hypotheses and
  include executable examples. Existing finite-sample normalizations, including
  `10*Z_A-32`, have been retained.
- Storage lookup tests were updated to expect the configuration already included
  by the current `DistributionKey` implementation; no storage algorithm changed.

## Hypothesis and scale conventions

Write `m_r = mean((x-mean(x))**r)`, `s0=sqrt(m2)`,
`s1=sqrt(n*m2/(n-1))`, and `z0=(x-mean(x))/s0`, `z1=(x-mean(x))/s1`.
Changing `ddof` changes a finite-sample statistic, even if the difference
vanishes asymptotically. The tests explicitly retain these distinctions.

Only KS and CVM test a fully specified normal law; their `mean` and `var` must
be supplied, and `var` means variance. The other generated EDF and all frequency
criteria estimate the mean and **unbiased** variance on each sample. Thus
D+ and D− generated classes have a fitted hypothesis, whereas legacy two-sided
KS has a fixed hypothesis. Lilliefors is the fitted two-sided KS statistic.

Fifty-two classes have a family hypothesis `{}`. For location-scale-invariant
settings their null can be simulated with N(0,1), repeating all fits. This
representative distribution does not fix parameters in the hypothesis.
Six graph classes instead fix variance and leave mean unknown. User-supplied
frequency bin edges are in original units and can break scale invariance.

For fitted normal EDF tests, independent uniform CDF values cannot replace
refitted normal samples. Neither the known-CDF EDF tables nor the familiar
multinomial chi-square law can be transferred automatically. In particular,
`chi-square(k-3)` is not asserted for raw-data estimates with adaptive bins.
For two-tail tests simulate both quantiles; a signed statistic's null need not
be symmetric. Directional KS deviations are nonnegative and reject in the
**upper** tail.

## Sources and level of verification

The following primary texts were inspected where accessible:

- [Zhang (2002), §§2–3](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf): EDF
  functionals, including half-rank and quarter-rank corrections. KS here is D,
  not the squared distance printed in §2.1.
- [Doornik and Hansen, author manuscript, appendix p.7](https://doornik.com/research/normal2.pdf):
  skewness and conditional-kurtosis transformations; rounded printed examples
  provide an external numerical check. The journal version is
  [Doornik–Hansen (2008)](https://doi.org/10.1111/j.1468-0084.2008.00537.x).
- [Bonett and Seier (2002), equations (5)–(6)](https://sci2s.ugr.es/keel/pdf/specific/articulo/Normalidad4.pdf):
  `ddof=0`, the factor `sqrt(n+2)` and the two-sided rule. This is the individual
  Geary-kurtosis transform, not the paper's further combined test.
- [Bontemps and Meddahi (2005), §§2–3](https://www.olsendata.com/data_products/client_papers/papers/200409-BontempsMedahi-TestNormGMMApproach.pdf):
  orthonormal Hermite moments. This implementation uses iid weighting and
  `s1`; it does not implement the paper's HAC extension to dependent data.
- [Brys, Hubert and Struyf, author manuscript, §3 and Table 1](https://wis.kuleuven.be/stat/robust/papers/2008/Bryshubertstruyf-comp.pdf):
  the MC-LR quadratic form. The library uses more digits in the constants than
  the table prints; exact agreement with those additional digits is not claimed.
- [Cabaña and Cabaña (2003), author-posted full text](https://www.researchgate.net/publication/263579544_Tests_of_Normality_Based_on_Transformed_Empirical_Processes):
  truncated Hermite processes, fixed truncation five and continuous supremum.
- [Desgagné, Lafaye de Micheaux and Leblanc (2013), author copy](https://www.biostatisticien.eu/textes/desgagne2013.pdf):
  fitted-score approach. Independent quadrature verifies the implemented
  score means and covariance projection, within the precision of rounded constants.
- [Lemeshko et al. (2014), equations (1)–(3), (6)](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf):
  Kuiper normalizations and Watson centering. These equations alone do not
  validate calibration after normal parameter fitting.

Direct scientific references for all 60 statistics appear in the coverage
table below; their individual pages provide further bibliographic detail. A bibliographic link alone is not evidence of a line-by-line formula
check. Original full texts for several older methods were inaccessible;
those methods were checked against independent mathematical constructions or
external software, with approximations stated explicitly below.

External software checks use the official SciPy definitions of
[Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html),
[skewness](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html),
[kurtosis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html)
and [Filliben plotting positions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html).
Agreement with a numerical implementation complements, but does not replace,
scientific attribution.

## Formula-by-formula coverage

Each row names one concrete statistic by its unique short code. Follow the
page link for its full formula and input restrictions. The scientific-source
column links directly to articles or technical reports. Where attribution is
unresolved, the column identifies related literature and its limits explicitly.
“Local” means that the mathematical functional is checked but its exact
published test/calibration has not been established. “Approximation” identifies
a finite-sample or numerical approximation; it is not an exact distributional claim.

| Code and page | Definition checked / independent evidence | Qualification | Scientific source |
| --- | --- | --- | --- |
| [`BHS`](bhs_normality_criterion.md) | n times the MC/LMC/RMC quadratic form; exact enumeration of medcouple kernels and strict median halves | Rounded asymptotic covariance; ties use an explicit convention | [Brys, Hubert & Struyf (2008)](https://doi.org/10.1007/s00180-007-0083-7) |
| [`BS`](bonett_seier_normality_criterion.md) | sqrt(n+2)*(13.29*log(s0/mean(abs(x-mean(x))))-3)/3.54; original eqs. (5)–(6) | Geary-kurtosis component; both tails | [Bonett & Seier (2002)](https://doi.org/10.1016/S0167-9473(02)00074-9) |
| [`BM1`](bontemps_meddahi1_normality_criterion.md) | Sum of squared normalized Hermite means for orders 3–4; scipy.special evaluation | s1 finite-sample convention; iid only | [Bontemps & Meddahi (2005)](https://doi.org/10.1016/j.jeconom.2004.02.014) |
| [`BM2`](bontemps_meddahi2_normality_criterion.md) | Same construction for orders 3–6; scipy.special evaluation | s1 finite-sample convention; iid only | [Bontemps & Meddahi (2005)](https://doi.org/10.1016/j.jeconom.2004.02.014) |
| [`CC1`](cabana_cabana1_normality_criterion.md) | Continuous supremum of the skewness-focused Hermite process; scalar optimization versus polynomial roots | Truncation five; no claim of consistency against every alternative | [Cabaña & Cabaña (2003)](https://doi.org/10.1023/A:1026235220018) |
| [`CC2`](cabana_cabana2_normality_criterion.md) | Continuous supremum of the kurtosis-focused Hermite process; scalar optimization | Truncation five | [Cabaña & Cabaña (2003)](https://doi.org/10.1023/A:1026235220018) |
| [`CS`](chen_shapiro_normality_criterion.md) | sqrt(n)*(1-mean(normalized adjacent spacings)/s1) | Approximate normal scores | [Chen & Shapiro (1995)](https://doi.org/10.1080/00949659508811711) |
| [`COIN`](coin_normality_criterion.md) | Squared cubic OLS coefficient on normal order scores; independent order-density quadrature and least squares | Approximation: nscor2, 4<=n<=2000; original finite-sample implementation not fully certified | [Coin (2008)](https://doi.org/10.1016/j.csda.2007.07.012) |
| [`DAP`](dagostino_pearson_normality_criterion.md) | Z_skew²+Z_kurtosis²; scipy.stats.normaltest | Upper tail, not signed skewness | [D’Agostino & Pearson (1973)](https://doi.org/10.1093/biomet/60.3.613) |
| [`D`](dagostino_normality_criterion.md) | Normalized rank-weighted contrast; independent pairwise absolute-distance representation | Rounded asymptotic centering/scaling; both tails | [D’Agostino (1971)](https://doi.org/10.1093/biomet/58.2.341) |
| [`DLDMZEPD`](desgagne_lafaye_normality_criterion.md) | n*r.T*A*r; quadrature of score means, covariance and scale-score projection | Ill-conditioned matrix and rounded coefficients; rel tolerance 2e-4 | [Desgagné, Lafaye de Micheaux & Leblanc (2013)](https://doi.org/10.1080/03610926.2011.577548) |
| [`DH`](doornik_hansen_normality_criterion.md) | Squared transformed skewness plus conditional kurtosis; published appendix numerical example | n>=8; finite-sample chi-square approximation is not exact | [Doornik & Hansen (2008)](https://doi.org/10.1111/j.1468-0084.2008.00537.x) |
| [`EP`](epps_pulley_normality_criterion.md) | Gaussian-weighted empirical-characteristic-function distance; numerical integration | z0 standardization | [Epps & Pulley (1983)](https://doi.org/10.1093/biomet/70.3.723) |
| [`FILLI`](filliben_normality_criterion.md) | Correlation with approximate normal order medians; scipy.stats.probplot | Approximate plotting positions | [Filliben (1975)](https://doi.org/10.1080/00401706.1975.10489279) |
| [`GMG`](gmg_normality_criterion.md) | s0 divided by sqrt(pi/2)*mean(abs(x-median(x))) | Directed heavy-tail test; upper tail | [Gel, Miao & Gastwirth (2007)](https://doi.org/10.1016/j.csda.2006.08.022) |
| [`GLB`](glen_leemis_barr_normality_criterion.md) | Beta order-CDF transformation and reversed-weight logarithmic sum; binomial-tail identity | Local fitted-normal functional; exact author attribution unconfirmed | [Glen, Leemis & Barr (2001)](https://doi.org/10.1109/24.963129); exact attribution unconfirmed |
| [`AVGDEGREE`](graph_average_degree_normality_criterion.md) | mean degree; exhaustive graph construction/subset enumeration | Local radius range/(10*m2); known variance; both tails | [Ebner, Nestmann & Schulte](https://arxiv.org/abs/1812.08638): related uniformity research only; exact source not established |
| [`CLIQUENUMBER`](graph_clique_number_normality_criterion.md) | maximum clique size; exhaustive graph construction/subset enumeration | Local radius range/(10*m2); known variance; both tails | [Ebner, Nestmann & Schulte](https://arxiv.org/abs/1812.08638): related uniformity research only; exact source not established |
| [`CONNECTEDCOMPONENTS`](graph_connected_components_normality_criterion.md) | number of components; exhaustive graph construction/subset enumeration | Local radius range/(10*m2); known variance; both tails | [Ebner, Nestmann & Schulte](https://arxiv.org/abs/1812.08638): related uniformity research only; exact source not established |
| [`EDGESNUMBER`](graph_edges_number_normality_criterion.md) | edge count; exhaustive graph construction/subset enumeration | Local radius range/(10*m2); known variance; both tails | [Ebner, Nestmann & Schulte](https://arxiv.org/abs/1812.08638): related uniformity research only; exact source not established |
| [`INDEPENDENCENUMBER`](graph_independence_number_normality_criterion.md) | maximum independent-set size; exhaustive graph construction/subset enumeration | Local radius range/(10*m2); known variance; both tails | [Ebner, Nestmann & Schulte](https://arxiv.org/abs/1812.08638): related uniformity research only; exact source not established |
| [`MAXDEGREE`](graph_max_degree_normality_criterion.md) | maximum degree; exhaustive graph construction/subset enumeration | Local radius range/(10*m2); known variance; both tails | [Ebner, Nestmann & Schulte](https://arxiv.org/abs/1812.08638): related uniformity research only; exact source not established |
| [`HOSKING1`](hosking1_normality_criterion.md) | Quadratic ratios of L-moments with trim 0; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`HOSKING2`](hosking2_normality_criterion.md) | Quadratic ratios of L-moments with trim 1; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`HOSKING3`](hosking3_normality_criterion.md) | Quadratic ratios of L-moments with trim 2; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`HOSKING4`](hosking4_normality_criterion.md) | Quadratic ratios of L-moments with trim 3; exhaustive subsample order-statistic averaging | Local piecewise constants; no universal chi-square calibration | [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x); [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002): moment construction, not local calibration constants |
| [`JB`](jarque_bera_normality_criterion.md) | n*(skew²/6+excess_kurtosis²/24); scipy.stats.jarque_bera | Asymptotic calibration requires care at small n | [Jarque & Bera (1987)](https://doi.org/10.2307/1403192) |
| [`KURTOSIS`](kurtosis_normality_criterion.md) | Signed Anscombe–Glynn cube-root transform; scipy.stats.kurtosistest | n>=5; approximate calibration, especially below 20 | [Anscombe & Glynn (1983)](https://doi.org/10.1093/biomet/70.1.227) |
| [`LG`](looney_gulledge_normality_criterion.md) | Correlation with Blom normal scores; independent correlation matrix | Approximate plotting positions | [Looney & Gulledge (1985)](https://doi.org/10.1080/00031305.1985.10479395) |
| [`MI`](martinez_iglewicz_normality_criterion.md) | Classical median-centered scale divided by truncated biweight scale; direct outlier calculation | Positive MAD and nonzero biweight denominator required | [Martinez & Iglewicz (1981)](https://doi.org/10.1093/biomet/68.1.331) |
| [`RJB`](robust_jarque_bera_normality_criterion.md) | n*m3²/(6*J^6)+n*(m4/J^4-3)²/64; direct moments and median absolute deviations | Upper tail; J uses the median | [Gel & Gastwirth (2008)](https://doi.org/10.1016/j.econlet.2007.05.022) |
| [`RJ`](ryan_joiner_normality_criterion.md) | Normal-score correlation; independent plotting positions for every alpha and averaged tied ranks | Weighted/alpha options change calibration | [Ryan & Joiner (1976), technical report](https://www.additive-net.de/en/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner) |
| [`SF`](sf_normality_criterion.md) | Squared Blom-score correlation | Approximation to expected normal order statistics | [Shapiro & Francia (1972)](https://doi.org/10.1080/01621459.1972.10481232); [Weisberg & Bingham (1975)](https://doi.org/10.1080/00401706.1975.10489283) |
| [`SWRG`](swrg_normality_criterion.md) | Normalized score contrast squared; inverse Brownian-bridge quantile covariance calculation | Approximate order-score covariance; lower tail | [Rahman & Govindarajulu (1997)](https://doi.org/10.1080/02664769723828) |
| [`SW`](shapiro_wilk_normality_criterion.md) | Royston approximation to Shapiro–Wilk W; SciPy at n=3,4,5,6,8,20,100,1000 | Coefficient approximation, not exact covariance inversion | [Shapiro & Wilk (1965)](https://doi.org/10.1093/biomet/52.3-4.591); [Royston (1995)](https://doi.org/10.2307/2986146) |
| [`SKEW`](skew_normality_criterion.md) | Signed D’Agostino transform; scipy.stats.skewtest and exact zero-skew regression | n>=8; both tails | [D’Agostino (1970)](https://doi.org/10.1093/biomet/57.3.679) |
| [`SH`](spiegelhalter_normality_criterion.md) | Power combination of standardized range and absolute deviations; direct powers versus log-space code | Documented symmetric-alternative functional; original full text not inspected | [Spiegelhalter (1977)](https://doi.org/10.1093/biomet/64.2.415) |
| [`ZQ`](zhang_q_normality_criterion.md) | log of mean endpoint spacings divided by mean lag-four spacings | n>=8; both tails; exact original text not inspected | [Zhang (1999)](https://doi.org/10.1080/02664769922395) |
| [`ZQS`](zhang_q_star_normality_criterion.md) | ZQ applied to the reflected sample; independent spacing expression | n>=8; both tails | [Zhang (1999)](https://doi.org/10.1080/02664769922395) |
| [`ZWA`](zhang_wu_a_normality_criterion.md) | 10*Zhang_A-32; density-quadrature CDF and independent rank sum | Historical affine normalization, n>=4 | [Zhang & Wu (2005)](https://doi.org/10.1016/j.csda.2004.05.034) |
| [`ZWC`](zhang_wu_c_normality_criterion.md) | Zhang_C log-odds discrepancy; independent rank sum | Same functional as generated Zhang C, n>=4 | [Zhang & Wu (2005)](https://doi.org/10.1016/j.csda.2004.05.034) |
| [`KS`](kolmogorov_smirnov_normality_criterion.md) | max(D+,D−) against specified normal CDF; scipy.stats.kstest in all directions | Fixed mean/variance, n>=1 | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256) |
| [`AD`](anderson_darling_normality_criterion.md) | Weighted EDF integral; density and EDF quadrature, using z1 | Fitted normal, no finite-sample multiplier | [Anderson & Darling (1954)](https://doi.org/10.1080/01621459.1954.10501232); [Stephens (1976)](https://doi.org/10.1214/aos/1176343411) |
| [`CVM`](cvm_normality_criterion.md) | Integral of squared EDF discrepancy; independent quadrature and scipy.stats.cramervonmises | Fixed normal; legacy full code CVM_GOODNESS_OF_FIT | [Cramér (1928)](https://doi.org/10.1080/03461238.1928.10416872) |
| [`LILLIE`](lilliefors_normality_criterion.md) | max(D+,D−) after mean/s1 fitting; independently refitted KS | Fitted-normal tables, not known-CDF KS tables | [Lilliefors (1967)](https://doi.org/10.1080/01621459.1967.10482916) |
| [`KS_PLUS`](kolmogorov_smirnov_d_plus_normality_criterion.md) | max(i/n-u_i); independent density quadrature | Fitted normal; upper rejection tail | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256); [Lilliefors (1967)](https://doi.org/10.1080/01621459.1967.10482916) (parameter-fitting context) |
| [`KS_MINUS`](kolmogorov_smirnov_d_minus_normality_criterion.md) | max(u_i-(i-1)/n); independent density quadrature | Fitted normal; upper rejection tail | [Smirnov (1948)](https://doi.org/10.1214/aoms/1177730256); [Lilliefors (1967)](https://doi.org/10.1080/01621459.1967.10482916) (parameter-fitting context) |
| [`LIAO_SHIMOKAWA`](liao_shimokawa_normality_criterion.md) | Sum of local EDF maxima weighted by 1/sqrt(u*(1-u)*n) | Normal-family application of a general functional; original paper concerns other families | [Liao & Shimokawa (1999)](https://doi.org/10.1080/00949659908811965) |
| [`WATSON`](watson_normality_criterion.md) | Centered EDF squared integral | Fitted normal | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109) |
| [`KUIPER`](kuiper_normality_criterion.md) | D++D− | Fitted normal, unscaled | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60)50006-0) |
| [`KUIPER_SCALED`](kuiper_scaled_normality_criterion.md) | sqrt(n)*(D++D−) | Fitted normal | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60)50006-0); [Lemeshko et al. (2014)](https://doi.org/10.3103/S875669901401004X) |
| [`KUIPER_CORRECTED`](kuiper_corrected_normality_criterion.md) | (sqrt(n)+0.155+0.24/sqrt(n))*(D++D−) | Normalization does not justify known-CDF calibration | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60)50006-0); [Lemeshko et al. (2014)](https://doi.org/10.3103/S875669901401004X) |
| [`ZHANG_ZK`](zhang_k_normality_criterion.md) | Maximum rank-weighted binary log-likelihood discrepancy | Fitted normal | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [`ZHANG_ZA`](zhang_a_normality_criterion.md) | Reciprocal-rank weighted logarithmic tails | Fitted normal; no 10*A-32 scaling | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [`ZHANG_ZC`](zhang_c_normality_criterion.md) | Squared log-odds discrepancies with quarter-rank correction | Fitted normal, n>=2 | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [`CHI2_PEARSON`](chi2_pearson_normality_criterion.md) | Power divergence lambda=1; independent scipy.stats.power_divergence | Fitted mean/s1 and custom bins | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`FREEMAN_TUKEY`](freeman_tukey_normality_criterion.md) | Power divergence lambda=-1/2, including factor 4 | Fitted mean/s1 and custom bins | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`MODIFIED_LOG_LIKELIHOOD`](modified_log_likelihood_normality_criterion.md) | Power divergence lambda=-1; explicit empty-cell limit | Positive infinity at zero observed count | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`CHI2_NEYMAN`](neyman_chi2_normality_criterion.md) | Power divergence lambda=-2; explicit empty-cell limit | Positive infinity at zero observed count | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [`CRESSIE_READ`](cressie_read_normality_criterion.md) | Power divergence lambda=2/3, including factor 9/5 | Zero counts contribute zero | [Cressie & Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |

## Legacy definitions and unresolved scientific attribution

**Hosking 1–4.** The L-moment and trimmed L-moment calculations agree with
exhaustive averaging of order-statistic combinations. The three historical
rows `(mu4,v3,v4)` for `n<=25`, `25<n<=50` and `n>50` are not an asymptotically
shrinking covariance sequence. Neither [Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x)
nor [Hosking (2007)](https://doi.org/10.1016/j.jspi.2006.12.002) is claimed to
justify these exact rows. These are local discrepancies: simulate at the actual
sample size. No unsupported replacement constants were introduced.

**Graph statistics.** Exhaustive small-graph checks establish their combinatorial
values. The sample-dependent radius `range/(10*m2)` changes as inverse scale,
while interpoint distances change as scale. Hence location invariance holds,
but scale invariance does not. Mean can be set to zero for simulation; variance
must match the hypothesis. No paper establishing this exact normality procedure
was located. [Ebner, Nestmann and Schulte](https://arxiv.org/abs/1812.08638)
is explicitly cited as related graph-testing research about **uniformity**, not
as provenance for these six normality tests.

**GLB.** The fitted beta transformation and reversed logarithmic weights are
reproduced using binomial probabilities. This does not make the expression the
ordinary Anderson–Darling statistic. The [Glen–Leemis–Barr paper](https://doi.org/10.1109/24.963129)
was identified, but the primary full text was inaccessible. Its exact equality
to this fitted-normal implementation remains unconfirmed. Use simulation of the
implemented functional, not an assumed GLB table.

**Approximations and directed tests.** Coin's regression is checked against
quadrature of expected normal order statistics, with tolerance reflecting its
`nscor2` approximation. Bonett–Seier is a kurtosis component; GMG is directed
against heavy tails. Fixed-order Hermite and moment tests are not proved
consistent against every nonnormal alternative by this audit. A finite value
or Monte Carlo calibration alone does not establish omnibus power.

## Numerical verification and reproducibility

Result: **1495 tests passed** in the isolated focused run. Ruff checks passed.
`mkdocs build --site-dir /tmp/pysatl-normal-docs-audit` succeeded; its warnings
concern existing broken links outside the normality section. All normality
pages, examples, catalog entries and local links passed the dedicated checks.

The focused suite consists of:

- `tests/distributions/test_normal.py`
- `tests/distributions/test_normal_parameters.py`
- `tests/distributions/test_normal_hypotheses.py`
- `tests/distributions/test_normal_correctness.py`
- `tests/distributions/test_normal_mathematical_validation.py`
- `tests/distributions/test_normal_documentation.py`
- `tests/statistics/test_normal_dynamic.py`
- `tests/statistics/test_normal_numba.py`
- `tests/statistics/test_normal_stub.py`

The new numerical references do not call the production EDF/frequency kernels.
They integrate the normal density and EDF discrepancies, use SciPy multinomial
power divergences, calculate score correlations and robust scales, invert a
quantile covariance matrix, and integrate order-statistic densities. Existing
checks independently optimize the Cabaña supremum, enumerate trimmed L-moments,
check all six graph functionals and integrate the fitted-score covariance.

Cases include normal and skewed samples, repeated values, symmetric samples,
small-size boundaries, refitting across calls, nonmutation, invalid and masked
data, signed tails, and positive affine transformations. Location-scale-invariant
family cases are tested at scales `1e-250` and `1e250`; graph statistics and
fixed-boundary binning are not incorrectly required to have this invariance.
Neyman and modified likelihood legitimately return infinity at empty observed
cells with positive expected probability. Extremely small probabilities may
still be limited by floating-point representability.

EDF comparisons use `rel=2e-9, abs=2e-10`; SciPy Shapiro comparisons use
`abs=2e-8`; the Coin approximation uses `rel=3e-3, abs=1e-6`; the rounded
Desgagné covariance comparison uses `rel=2e-4`. These are bounds for the checked
cases, not global error guarantees.

Ordinary pytest collection is currently blocked by the pre-existing import of
`AbstractChi2Statistic` from `edf.py` in `inverse_gamma.py`. Focused verification
loads the real normal, EDF, frequency, distribution and binning modules while
bypassing only the unrelated `continuous/__init__.py` import aggregator. This
is not evidence that package initialization is healthy. To reproduce that run
from the repository root:

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
    "tests/distributions/test_normal.py",
    "tests/distributions/test_normal_parameters.py",
    "tests/distributions/test_normal_hypotheses.py",
    "tests/distributions/test_normal_correctness.py",
    "tests/distributions/test_normal_mathematical_validation.py",
    "tests/distributions/test_normal_documentation.py",
    "tests/statistics/test_normal_dynamic.py",
    "tests/statistics/test_normal_numba.py",
    "tests/statistics/test_normal_stub.py", "-q",
]))
PY
```

## Numba optimization verification

Numerical kernels use static class methods decorated with
`@jit(nopython=True, cache=True)`, without fast-math reassociation. The public
entry points retain input validation and stable rescaling. All 60 criteria
reach compiled kernels; generated EDF and frequency criteria reuse their
existing common kernels. Normal quantiles and beta/log-tail functions remain
in SciPy, and Cabaña polynomial root finding remains in NumPy. Compiling a
reduction does not mean the entire public call is compiled.

The 128 additional checks in `test_normal_numba.py` exercise every public path
on continuous and tied data, require actual nopython signatures and disk-cache
configuration, and compare compiled results with the Python kernel bodies.
They also check large Hosking combination weights and strict graph-radius
boundaries against explicit graph calculations. Independent formula checks
listed above remain the mathematical reference. Baseline comparisons before
and after optimization also passed on normal samples of sizes 12, 100 and 1000
for all 39 explicitly implemented statistics (`rtol=2e-8`, `atol=2e-9`).

Graph statistics now use sorted scans instead of constructing adjacency lists;
the strict edge condition remains `abs(x_i-x_j) < radius`. BHS uses one flat
medcouple-kernel array instead of multiple broadcast matrices, preserving its
tie convention. Hosking uses floating-point low-order combination products to
avoid fixed-width integer overflow. Coin and Shapiro–Wilk evaluate quantiles
in batches, and Cabaña reuses its fixed polynomial basis.

A local warmed benchmark used the public API, `n=200`, seed 123 and the minimum
of three batches of ten calls, comparing against the pre-optimization source.
Observed speedups were approximately 9× for Epps–Pulley, 13–15× for Hosking
2–4, 15× for Jarque–Bera, 41× for Coin and 89× for Shapiro–Wilk. The four graph
statistics that previously built adjacency lists improved by about 186–222×.
GLB was essentially unchanged (0.98×), since SciPy beta tails dominate it.
These are local measurements, not performance guarantees; compilation/cache
loading is excluded from these speedup ratios.

To measure first-call and warmed execution separately for all 60 criteria:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_normal_pipeline.py --isolated-imports
```

The explicit isolation flag bypasses the unrelated package-import failure
described above. Once that failure is fixed, omit the flag. First-call timings
include compilation or disk-cache loading, depending on the cache state.

## Package layout and compatibility

Like `continuous/uniform`, the normality implementation is a package:

- `continuous/normal/normal_statistics.py` contains the statistics and factories.
- `continuous/normal/__init__.py` explicitly re-exports the 60 concrete classes
  and two abstract bases, retaining their public `continuous.normal` identities.
- `continuous/normal/__init__.pyi` describes the complete public interface.

Existing `from ...continuous.normal import ...` imports and pickle references
retain their paths. The 65 stub/package checks cover exports, method signatures,
class identity, old pickle references and unique criterion discovery. Generate
and verify the package stub with `scripts/generate_normal_stub.py` and its
`--check` option; generation reads the source without importing the package.
Mypy checking of the stub passed with `--follow-imports=silent`.

## Documentation coverage

The catalog and navigation contain all 60 classes, with exactly one individual
page per class, plus the catalog and this report. Tests compare class names,
codes and exports with documentation, require examples and scientific links,
execute every Python example and check local Markdown links. This catches a
page accidentally documenting a different criterion, as happened with DAP.
All concrete class docstrings include references and executable examples.

An available related scientific link is explicitly distinguished from a source
that validates the exact implementation. The unresolved attribution of the
legacy methods above is part of the result, not concealed by coverage checks.
