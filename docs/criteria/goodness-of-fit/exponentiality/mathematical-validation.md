# Mathematical verification of exponential statistics

Audit date: **2026-10-15**. Scope: all **50 concrete classes** exported by
`continuous/exponent/exponent_statistics.py`, their shared kernels in `edf.py` and `frequency.py`,
and `monte_carlo_exponential`. There are 14 generated EDF classes, 5 generated
frequency classes and 31 specialized classes. See the [complete catalog](index.md).
Direct article links appear beside every criterion in the formula inventory;
full titles are collected in the [scientific bibliography](#scientific-bibliography).

## Result and outstanding findings

Every implemented scalar definition was compared with an independent reference
at three positive samples, in every supported fixed/unknown-rate mode: **195
comparisons covering 50 classes and 65 hypothesis modes**. These checks passed.
Additional tests cover endpoints, invalid inputs, extreme scales, per-call
fitting, exact arithmetic, integral representations and scientific reference
values. This is **not a blanket certification of all 50 historical names**.

| Finding | Consequence | Disposition |
|---|---|---|
| Hollander–Proschan previously rejected large values of the raw probability U-statistic | NBU alternatives make the probability smaller than 1/4 | Fixed: `LeftAlternative` rejects small values; numerical statistic unchanged |
| Gnedenko and Harris previously exposed only `RightAlternative` | Small F ratios could not reject the null | Fixed: `TwoSidedAlternative` uses both tails; scalar ratios and exact F laws unchanged |
| WE previously had an extra factor (n-1)/n | Numerical values differed from the published formula | Fixed: WE = G - 1/n; full Ascher text and historical tail convention not verified |
| Six range/10 graph criteria lack an identified primary source for this exact test | Graph algorithms are correct, but published exponentiality-test provenance is unestablished | Marked local constructions; cited graph literature is background only |
| Graph criteria and exponential Shapiro–Wilk W are translation invariant | Positive shifts of an exponential law cannot be detected | No omnibus zero-origin consistency claim |
| Infinite frequency penalties were rejected by the Monte Carlo helper | Neyman and modified log-likelihood could not be calibrated with sparse cells | Fixed: infinities remain in ranks; only NaNs are rejected; step quantiles avoid infinity interpolation |
| Package aggregate imports removed `AbstractChi2Statistic` from `edf.py` | Normal imports/pytest collection fail before reaching the exponential module | Pre-existing integration blocker; numerical tests and examples were run in an explicitly isolated package namespace |

## Hypotheses, fitting and invariance

The null family is $F_\lambda(x)=1-e^{-\lambda x}$ for $x\ge0$, $\lambda>0$,
with known origin zero. `lam` is a **rate**, with scale $1/\lambda$.
For unknown rate, $\widehat\lambda=1/\bar x$ and $y_i=x_i/\bar x$.

| Classes | Accepted parameters | Fitting/calibration |
|---|---|---|
| KS D+, D− and D | Fixed `lam` | No fit; continuous fixed-null probability transform |
| Lilliefors | Empty schema | Fit rate on every sample, including every replicate |
| Other 10 generated EDF classes | Fixed `lam` or empty schema | Fixed CDF or per-sample MLE |
| Five frequency classes | Fixed `lam` or empty schema | Bind specified/fitted distribution; preserve binning |
| 31 specialized classes | Empty schema | Scale-invariant definition; no fitted parameters stored |

Every concrete class declares its supported hypotheses. Omitted parameters
are not filled with defaults. The family base supplies no support declaration.
For fitted criteria, Exp(1) simulation is justified by positive-scale invariance,
not by treating the fitted sample as independent uniform observations.

`EqualProbabilityBins()` uses a sample-size-dependent count. Explicit counts,
count rules and binning strategies belong to the criterion configuration.
Fitted-rate frequency calibration with an arbitrary fixed set of raw-unit
edges is not scale invariant; the built-in helper rejects it. A known-rate
criterion may use those edges because simulation uses that specified rate.

## Formula inventory: every concrete class

The linked pages contain full class names, short codes, parameter settings,
input limits, examples and scientific links. Notation: $x_{(i)}$ are ordered
observations, $y_i=x_i/\bar x$, $u_i=F(x_{(i)})$, $r_i=i-1/2$,
$D^+=\max_i(i/n-u_i)$, $D^-=\max_i(u_i-(i-1)/n)$ and
$d_i=(n-i+1)(x_{(i)}-x_{(i-1)})$ with $x_{(0)}=0$.
$O_j$ and $E_j=np_j>0$ are aligned cell counts.

### Edf definitions

| Criterion | Implemented scalar | API rejection | Scientific source |
|---|---|---|---|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_exp_criterion.md) | $D^+=\max_i(i/n-u_i)$ | right tail | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_exp_criterion.md) | $D^-=\max_i(u_i-(i-1)/n)$ | right tail | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_exp_criterion.md) | $D=\max(D^+,D^-)$ | right tail | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [Lilliefors exponential](lilliefors_exp_criterion.md) | $D=\max(D^+,D^-),\quad u_i=1-e^{-x_{(i)}/\bar x}$ | right tail | [Lilliefors (1969)](https://doi.org/10.1080/01621459.1969.10500983) |
| [Anderson–Darling](anderson_darling_exp_criterion.md) | $-n-\frac1n\sum_i(2i-1)[\log u_i+\log(1-u_{n+1-i})]$ | right tail | [Anderson and Darling (1952)](https://doi.org/10.1214/aoms/1177729437) |
| [Cramér–von Mises](cvm_exp_criterion.md) | $W^2=\frac1{12n}+\sum_i[u_i-(i-1/2)/n]^2$ | right tail | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [Liao–Shimokawa](liao_shimokawa_exp_criterion.md) | $\frac1{\sqrt n}\sum_i\frac{\max(i/n-u_i,u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}$ | right tail | [Liao and Shimokawa (1999)](https://doi.org/10.1080/00949659908811965) — original models differ; recalibration required |
| [Watson](watson_exp_criterion.md) | $U^2=W^2-n(\bar u-1/2)^2$ | right tail | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109) |
| [Kuiper](kuiper_exp_criterion.md) | $V=D^++D^-$ | right tail | [Kuiper (1960)](https://doi.org/10.1016/S1385-7258(60) |
| [Scaled Kuiper](kuiper_scaled_exp_criterion.md) | $V_s=\sqrt n(D^++D^-)$ | right tail | [Lemeshko et al. (2014)](https://doi.org/10.3103/S875669901401004X) |
| [Corrected Kuiper](kuiper_corrected_exp_criterion.md) | $V_c=(\sqrt n+0.155+0.24/\sqrt n)(D^++D^-)$ | right tail | [Lemeshko et al. (2014)](https://doi.org/10.3103/S875669901401004X) |
| [Zhang Z_K](zhang_k_exp_criterion.md) | $\max_i\left[r_i\log\frac{r_i}{nu_i}+(n-r_i)\log\frac{n-r_i}{n(1-u_i)}\right],\quad r_i=i-1/2$ | right tail | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [Zhang Z_A](zhang_a_exp_criterion.md) | $-\sum_i\left[\frac{\log u_i}{n-i+1/2}+\frac{\log(1-u_i)}{i-1/2}\right]$ | right tail | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |
| [Zhang Z_C](zhang_c_exp_criterion.md) | $\sum_i\left[\log\frac{1-u_i}{u_i}-\log\frac{n-i+1/4}{i-3/4}\right]^2$ | right tail | [Zhang (2002)](https://doi.org/10.1111/1467-9868.00337) |

### Frequency definitions

| Criterion | Implemented scalar | API rejection | Scientific source |
|---|---|---|---|
| [Pearson chi-square](pearson_exp_criterion.md) | $\sum_j(O_j-E_j)^2/E_j$ | right tail | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [Freeman–Tukey](freeman_tukey_exp_criterion.md) | $4\sum_j(\sqrt{O_j}-\sqrt{E_j})^2$ | right tail | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [Modified log-likelihood](modified_log_likelihood_exp_criterion.md) | $2\sum_j E_j\log(E_j/O_j)$ | right tail | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [Neyman chi-square](neyman_chi2_exp_criterion.md) | $\sum_j(O_j-E_j)^2/O_j$ | right tail | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |
| [Cressie–Read](cressie_read_exp_criterion.md) | $\frac95\sum_j O_j[(O_j/E_j)^{2/3}-1]$ | right tail | [Cressie and Read (1984)](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x) |

### Specialized definitions

| Criterion | Implemented scalar | API rejection | Scientific source |
|---|---|---|---|
| [Epps–Pulley](epps_pulley_exp_criterion.md) | $\sqrt{48n}\left(n^{-1}\sum_i e^{-y_i}-\tfrac12\right)$ | both tails | [Epps and Pulley (1986)](https://doi.org/10.1111/j.2517-6161.1986.tb01403.x) |
| [Ahsanullah](ahsanullah_exp_criterion.md) | $n^{-3}\sum_{i,j,k}\left[\mathbf1\{\vert x_i-x_j\vert <x_k\}-\mathbf1\{2\min(x_i,x_j)<x_k\}\right]$ | right tail | [Volkova and Nikitin (2013/2015)](https://www.mathnet.ru/eng/znsl5652) — finite-sample convention qualified below |
| [Atkinson](atkinson_exp_criterion.md) | $\sqrt n\left\vert \left(n^{-1}\sum_i y_i^p\right)^{1/p}-\Gamma(1+p)^{1/p}\right\vert $ | right tail | [Mimoto and Zitikis (2008)](https://doi.org/10.14490/jjss.38.187) |
| [Cox–Oakes](cox_oakes_exp_criterion.md) | $n+\sum_i(1-y_i)\log y_i$ | both tails | [Henze and Meintanis (2005)](https://doi.org/10.1007/s001840400322) — review; score derived independently |
| [Deshpande](deshpande_exp_criterion.md) | $\frac{1}{n(n-1)}\sum_{i\ne j}\mathbf1\{x_i>b x_j\}$ | right tail | [Deshpande (1983)](https://doi.org/10.1093/biomet/70.2.514) |
| [Epstein spacing log contrast](epstein_exp_criterion.md) | $\frac{2n\{\log\bar d-n^{-1}\sum_i\log d_i\}}{1+(n+1)/(6n)}$ | right tail | [Epstein (1960)](https://doi.org/10.1080/00401706.1960.10489892) — finite-sample correction not verified |
| [Frozini EDF distance](frozini_exp_criterion.md) | $n^{-1/2}\sum_i\left\vert u_i-\frac{i-1/2}{n}\right\vert $ | right tail | [Henze and Meintanis (2005)](https://doi.org/10.1007/s001840400322) — review; original attribution not verified |
| [Gini](gini_exp_criterion.md) | $\frac{\sum_{i<j}\vert x_i-x_j\vert }{(n-1)\sum_i x_i}$ | both tails | [Gail and Gastwirth (1978)](https://doi.org/10.1111/j.2517-6161.1978.tb01048.x) |
| [Gnedenko F ratio](gnedenko_exp_criterion.md) | $\frac{r^{-1}\sum_{i=1}^{r}d_i}{(n-r)^{-1}\sum_{i=r+1}^{n}d_i}$ | both tails | [Harris (1976)](https://doi.org/10.1002/nav.3800230116) — related spacing test; not original source |
| [Harris F ratio](harris_exp_criterion.md) | $\frac{(2r)^{-1}\left(\sum_{i=1}^{r}d_i+\sum_{i=n-r+1}^{n}d_i\right)}{(n-2r)^{-1}\sum_{i=r+1}^{n-r}d_i}$ | both tails | [Harris (1976)](https://doi.org/10.1002/nav.3800230116) |
| [Hegazy–Green 1](hegazy1_exp_criterion.md) | $n^{-1}\sum_i\left\vert y_{(i)}+\log(1-i/(n+1))\right\vert ^1$ | right tail | [Hegazy and Green (1975)](https://doi.org/10.2307/2347090); [Ushakov and Ushakov — exponential variants](https://doi.org/10.1080/03610918.2021.2009868) |
| [Hegazy–Green 2](hegazy2_exp_criterion.md) | $n^{-1}\sum_i\left\vert y_{(i)}+\log(1-i/(n+1))\right\vert ^2$ | right tail | [Hegazy and Green (1975)](https://doi.org/10.2307/2347090); [Ushakov and Ushakov — exponential variants](https://doi.org/10.1080/03610918.2021.2009868) |
| [Hollander–Proschan](holander_proshan_exp_criterion.md) | $\frac{\sum_{i,j,k\ \mathrm{distinct}}\mathbf1\{x_i>x_j+x_k\}}{n(n-1)(n-2)}$ | left tail | [Hollander and Proschan (1972)](https://doi.org/10.1214/aoms/1177692466) — lower-tail NBU direction |
| [Kimber–Michael](kimber_michael_exp_criterion.md) | $\max_i\left\vert A(u_i)-A((i-1/2)/n)\right\vert ,\quad A(u)=2\arcsin(\sqrt u)/\pi$ | right tail | [Kimber (1985)](https://doi.org/10.1093/biomet/72.3.661) |
| [Kochar exponential scores](kochar_exp_criterion.md) | $\sqrt{108n/17}\frac{\sum_i J(i/(n+1))x_{(i)}}{\sum_i x_i},\quad J(u)=2(1-u)[1-\log(1-u)]-1$ | right tail | [Kochar (1985)](https://doi.org/10.1080/03610928508828919) |
| [Lorenz ordinate](lorenz_exp_criterion.md) | $\frac{\sum_{i=1}^{\lfloor np\rfloor}x_{(i)}}{\sum_i x_i}$ | right tail | [Gail and Gastwirth (1978)](https://doi.org/10.1080/01621459.1978.10480100) |
| [Moran log-mean contrast](moran_exp_criterion.md) | $\gamma+n^{-1}\sum_i\log y_i$ | both tails | [Mimoto and Zitikis (2008)](https://doi.org/10.14490/jjss.38.187) |
| [Pietra](pietra_exp_criterion.md) | $\frac{1}{2n}\sum_i\vert y_i-1\vert $ | both tails | [Gail and Gastwirth (1978)](https://doi.org/10.1080/01621459.1978.10480100) — inequality context; exact attribution not verified |
| [Shapiro–Wilk exponential W](shapiro_wilk_exp_criterion.md) | $\frac{n(\bar x-x_{(1)})^2}{(n-1)\sum_i(x_i-\bar x)^2}$ | both tails | [Shapiro and Wilk (1972)](https://doi.org/10.1080/00401706.1972.10488921); [Spinelli and Stephens (1987)](https://www.stat.cmu.edu/technometrics/80-89/VOL-29-04/v2904471.pdf) |
| [Rossberg integral contrast](rossberg_exp_criterion.md) | $n^{-1}\sum_k[H_n(x_k)-G_n(x_k)]$ | right tail | [Volkova (2009/2010)](https://doi.org/10.1007/s10958-010-9934-9) — finite-sample convention qualified below |
| [WE](we_exp_criterion.md) | $\frac{\sum_i(x_i-\bar x)^2}{(\sum_i x_i)^2}$ | right tail | [Ascher (1990)](https://doi.org/10.1080/03610929008830292); [later formula](https://doi.org/10.1080/03610918.2021.2009868) — published normalization; historical tail not verified |
| [Wong–Wong extremal quotient](wong_wong_exp_criterion.md) | $x_{(n)}/x_{(1)}$ | right tail | [Wong and Wong (1979)](https://doi.org/10.1007/BF01893465) |
| [Greenwood](greenwood_exp_criterion.md) | $G=\frac{\sum_i x_i^2}{(\sum_i x_i)^2}$ | both tails | [Arendarczyk, Kozubowski and Panorska (2022)](https://doi.org/10.1111/sjos.12520) |
| [Baringhaus–Henze empirical Laplace transform](baringhaus_henze_elt_exp_criterion.md) | $n\int_0^\infty[(1+t)L_n^\prime(t)+L_n(t)]^2e^{-at}\,dt,\quad L_n(t)=n^{-1}\sum_i e^{-ty_i}$ | right tail | [Baringhaus and Henze (1991)](https://www.ism.ac.jp/editsec/aism/pdf/043_3_0551.pdf) |
| [Baringhaus–Henze mean residual life](baringhaus_henze_bhc_exp_criterion.md) | $n\int_0^\infty\left[n^{-1}\sum_i(y_i-t-1)\mathbf1\{y_i>t\}\right]^2e^{-at}\,dt$ | right tail | [Ebner (2023)](https://arxiv.org/abs/2209.05153); [Baringhaus and Henze (2008)](https://doi.org/10.1016/j.spl.2007.09.060) |
| [Graph edge count](graph_edges_number_exp_criterion.md) | $\vert E\vert $ | both tails | [Dettmann and Georgiou (2016)](https://arxiv.org/abs/1411.3617) — graph background only; exact test has no identified source |
| [Graph maximum degree](graph_max_degree_exp_criterion.md) | $\max_v\deg(v)$ | both tails | [Dettmann and Georgiou (2016)](https://arxiv.org/abs/1411.3617) — graph background only; exact test has no identified source |
| [Graph average degree](graph_average_degree_exp_criterion.md) | $2\vert E\vert /n$ | both tails | [Dettmann and Georgiou (2016)](https://arxiv.org/abs/1411.3617) — graph background only; exact test has no identified source |
| [Graph connected components](graph_connected_components_exp_criterion.md) | $\#\{\text{connected components}\}$ | both tails | [Dettmann and Georgiou (2016)](https://arxiv.org/abs/1411.3617) — graph background only; exact test has no identified source |
| [Graph clique number](graph_clique_number_exp_criterion.md) | $\omega(G)$ | both tails | [Dettmann and Georgiou (2016)](https://arxiv.org/abs/1411.3617) — graph background only; exact test has no identified source |
| [Graph independence number](graph_independence_number_exp_criterion.md) | $\alpha(G)$ | both tails | [Dettmann and Georgiou (2016)](https://arxiv.org/abs/1411.3617) — graph background only; exact test has no identified source |

For Rossberg, $H_n(t)$ averages the indicator that the first spacing of a
three-observation subsample is below $t$; $G_n(t)$ averages the indicator that
a two-observation minimum is below $t$. Subsamples contain distinct indices,
but the final empirical integral reuses all original observations. Ahsanullah,
in contrast, deliberately uses a V-statistic including repeated indices.

## Independent derivations and numerical references

### EDF and frequency definitions

KS and its one-sided variants were checked against SciPy's continuous fixed-CDF
test; Lilliefors was checked against KS evaluated at the fitted exponential CDF.
Cramér–von Mises was compared with SciPy and with integration of the squared
step-CDF discrepancy. Anderson–Darling and Watson were independently integrated
on each probability interval, including both support tails. Kuiper and its two
multipliers were compared with independently computed one-sided deviations.

The shared Zhang definitions match equations (3.1)–(3.3) of
[Zhang (2002)](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf); independent
high-precision fixtures and direct scalar logarithms check rank offsets and
normalization. Exponential log-survival is $-\lambda x$, avoiding loss of the
upper tail when the CDF rounds to one. Liao–Shimokawa's functional is checked
by direct summation, but its original estimated-Weibull critical values do not
apply to fitted exponential observations. Finite-sample Kuiper multipliers do
not constitute an exact fitted-null p-value.

For frequencies, the five kernels agree with the power-divergence family at
parameters $1,-1/2,-1,-2,2/3$, independently evaluated by SciPy from histogram
counts. Empty-cell contributions are, respectively, $E_j$, $4E_j$, $+\infty$,
$+\infty$, and zero. Known-rate Pearson has an asymptotic chi-square(k-1) limit
under suitable cell conditions; fitting by ungrouped MLE with moving cells does
not justify automatically subtracting another degree of freedom.

### Scores, moments and spacings

Differentiating the profiled Weibull log likelihood at shape 1 gives
$n+\sum_i(1-y_i)\log y_i$, verifying the Cox–Oakes sign and factor. The test
compares that analytic score with a central finite difference of the likelihood.
Its raw value is not standardized; a conventional asymptotic standardization
multiplies it by $\sqrt{6/n}/\pi$.

Atkinson uses an absolute power-mean contrast. The accepted computational
range $p>-1$ is broader than the usual finite-variance asymptotic range
$p>-1/2$. Near zero its geometric-mean limit is used; $p=1$ is excluded as
degenerate. Normal calibration of an unstandardized raw value is unjustified.
The [Mimoto–Zitikis paper](https://www.jstage.jst.go.jp/article/jjss/38/2/38_2_187/_article)
is linked for provenance; full-text verification was unavailable in this audit.

Under an exponential sample, $P_i=X_i/\sum_jX_j$ has Dirichlet(1,...,1)
distribution. Direct Dirichlet moment calculations give

$$
E(G)=\frac2{n+1},\qquad
\operatorname{Var}(G)=\frac{4(n-1)}{(n+1)^2(n+2)(n+3)}.
$$

They also give Moran's exact mean $\log n-\psi(n)$ and variance
$\psi_1(1)/n-\psi_1(n)$ for the centered value used here. Integrating a
Beta(1,n-1) marginal gives the exact Pietra mean $(1-1/n)^n$, rather than its
asymptotic center $1/e$. Gini uses denominator $n-1$, with exact null mean
$1/2$ and variance $1/[12(n-1)]$. WE is exactly

$$
WE=G-\frac1n.
$$

For normalized ordered spacings, independence and exponentiality imply that
sums over disjoint groups are independent gamma variables. Their ratios of
means yield F(2r,2(n-r)) for Gnedenko and F(4r,2(n-2r)) for Harris. These
identities validate the scalar ratios independently of historical attribution.
Both criteria now use `TwoSidedAlternative`, with equal probabilities in the
lower and upper tails. Regression checks construct samples from prescribed
spacings and verify rejection for both small and large ratios, and acceptance
for a ratio of one, through Monte Carlo calibration. Epstein's logarithmic spacing
contrast and correction were checked algebraically, but its attribution and
purported finite-sample approximation are not certified.

### Characterizations, direction and graph limitations

For independent continuous observations, the raw Hollander–Proschan statistic
estimates $P(X>Y+Z)$. Under NBU,
$\bar F(y+z)\le\bar F(y)\bar F(z)$, hence

$$
P(X>Y+Z)\le\left[\int\bar F(y)\,dF(y)\right]^2=\frac14.
$$

Thus the original [NBU procedure](https://doi.org/10.1214/aoms/1177692466)
rejects small values. The API now uses `LeftAlternative`. Uniform(1,2)
supplies a concrete regression case: its statistic is zero. A deterministic
sample in (1,2) checks the scalar and rejection through lower-tail Monte Carlo
calibration. The former upper-tail rule missed this NBU departure.
Deshpande's raw expectation $1/(1+b)$ follows by integrating
$P(X>bY)$; its upper-tail choice is compatible with the IFRA alternative.

Ahsanullah and Rossberg were checked against exhaustive indicator enumeration,
including repeated-index versus distinct-index conventions and strict ties.
Characterization identities alone do not establish consistency of a particular
signed integral against every alternative. Their exact original finite-sample
conventions remain qualified, rather than assigning unverified normal laws.

The six graph algorithms were compared with explicit NetworkX graphs, maximal
cliques and complement cliques. Their proximity threshold is data range/10.
Translation leaves every edge unchanged. Shapiro–Wilk W also cancels a shift
in both numerator and denominator, as discussed by
[Spinelli and Stephens (1987), §3](https://www.stat.cmu.edu/technometrics/80-89/VOL-29-04/v2904471.pdf).
Neither construction can consistently detect a positive location shift while
the declared origin is fixed at zero. The graph reference on each page is
background, not evidence for this exact exponentiality procedure.

### Transform criteria

The ELT definition was compared with the normalized differential-equation
construction in [Baringhaus and Henze (1991), pp. 552–553](https://www.ism.ac.jp/editsec/aism/pdf/043_3_0551.pdf).
Independent numerical quadrature verifies the implemented pair sum. For the
mean-residual-life family, direct integration and the independent pair
representation in [Ebner (2023), equations (4)–(5)](https://arxiv.org/html/2209.05153v2)
verify the piecewise implementation. Both keep tuning fixed; they refit the
mean for every sample. The accepted $0<a\le100$ range is an implementation
constraint, not a statement of the entire theoretical domain.

## Scientific provenance: what was and was not verified

Full text was available for the Baringhaus–Henze ELT, Ebner residual-life,
Zhang and Spinelli–Stephens comparisons above. Other linked originals were
checked at bibliographic/abstract level where accessible; this is not claimed
to be equation-by-equation verification of every paywalled article. Original
links for Deshpande, Gini, Lorenz, Kimber–Michael, Kochar, Wong–Wong, Ahsanullah
and Rossberg are recorded on their individual pages. No claim is made that
all external hosts are currently reachable or that their tables match this API.

The exponential HG1 and HG2 tests are described by Ushakov and Ushakov in the scientific article cited below. Their scientific attribution is established; they are not listed as tests with unconfirmed provenance. Calibrate under the fitted exponential null. Historical publication priority is separate from verification of the implemented formula and calibration. See [Ushakov and Ushakov, On sensitivity of exponentiality tests to data rounding: a Monte Carlo simulation study](https://doi.org/10.1080/03610918.2021.2009868).

Frozini's original
reference is a book chapter; a scientific comparison article is linked as
context, not misrepresented as its first publication. Cox–Oakes similarly
has a historical book source, with an independent likelihood derivation here.
WE is a published test name associated with Ascher (1990), not an invented local name.
The later rounding study cited below gives the definition

$$
WE_{\mathrm{published}}=\frac{\sum_i(x_i-\bar x)^2}{(\sum_i x_i)^2}=G-\frac1n.
$$

The implementation now returns this published formula. The former extra
factor $(n-1)/n$ has been removed. WE is an increasing affine function of
Greenwood for fixed sample size, but this
class currently uses the right tail while Greenwood uses both tails.
The full text of Ascher was not accessible during this follow-up; the formula
comparison uses the later article, which explicitly cites Ascher for WE.
The historical rejection-tail convention has not been verified here.

Sources: [Ascher (1990), A survey of tests for exponentiality, 19(5), 1811–1825](https://doi.org/10.1080/03610929008830292); [On sensitivity of exponentiality tests to data rounding: a Monte Carlo simulation study](https://doi.org/10.1080/03610918.2021.2009868).

The six graph classes still have no identified primary publication for their
exact range/10 tests.

## Numerical and Monte Carlo evidence

The combined isolated run passed **837 tests** (one upstream SciPy deprecation
warning). All **51 documentation examples** executed. Ruff and generated-stub
synchronization checks passed. MkDocs built successfully with no warnings for
`exponentiality/`; existing broken-link warnings remain in other families.

The audit runs the existing exponential correctness, numerical-boundary,
verification and generated-class suites, plus:

- `test_exponent_mathematical_audit.py`: all 50 scalar definitions, every
  supported rate mode, profile-likelihood score, infinity/NaN rank behavior,
  and an explicit Hollander–Proschan lower-tail rejection regression.
- `test_exponent_documentation.py`: one page per concrete class, complete
  catalog and MkDocs navigation, scientific links, local-link existence and
  execution of all 51 page/catalog examples.
- `test_edf.py`: shared-kernel reference integrals and high-precision Zhang
  values, including endpoint behavior.

The opt-in `test_exponential_validation.py` run passed **3 statistical tests**.
It checks signed-contrast normal approximations, exact spacing F laws, known-rate
KS/Pearson laws, Baringhaus–Henze null behavior, held-out power diagnostics and
legacy-class level diagnostics. Seeds are 20261008, 19461991 and 31082026;
the null-law stage uses up to 5000 replicates. These simulations do not replace
formula proofs, do not certify every alternative, and are not an all-50
uniform-power study. Discrete graph tests can be conservative.

The Monte Carlo helper now retains positive/negative infinity in comparisons.
It rejects NaNs instead of dropping replicates. For the upper tail,
$p=(1+\#\{T_b\ge T_{obs}\})/(B+1)$; two-tail values double the smaller
corrected tail probability, capped at one. Empirical inverse-CDF quantiles
use order statistics rather than interpolating infinity. Rejection is based
on the rank p-value, not on strict comparison with the displayed quantile.
The rank denominator, inclusive ties and use of the specified simulation rate
are separately tested. No claim is made that rounded or censored observations
follow the continuous-null calibration.

## Reproducing the isolated run

Normal pytest currently fails during collection because the aggregate
`continuous/__init__.py` imports removed legacy names from `edf.py`. This
pre-existing issue is outside the exponential formula/doc audit. To reproduce
this audit without replacing algorithms or mocking results, create only the
parent package namespace and load the real modules from disk:

```bash
.venv/bin/python - <<'PYTEST'
import sys
import types
from pathlib import Path
import pytest

root = Path.cwd()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
name = "pysatl_criterion.statistics.goodness_of_fit.continuous"
package = types.ModuleType(name)
package.__path__ = [str(root / "src/pysatl_criterion/statistics/goodness_of_fit/continuous")]
sys.modules[name] = package
raise SystemExit(pytest.main([
    "tests/distributions/test_exponent.py",
    "tests/distributions/test_exponent_correctness.py",
    "tests/distributions/test_exponent_numerics.py",
    "tests/distributions/test_exponential_verification.py",
    "tests/statistics/test_exponent_dynamic.py",
    "tests/statistics/test_exponent_mathematical_audit.py",
    "tests/statistics/test_exponent_documentation.py",
    "tests/statistics/test_edf.py", "-q",
]))
PYTEST
```

For the statistical run, set `PYSATL_STATISTICAL=1` and use
`tests/statistical/test_exponential_validation.py` in the same runner. An
isolated pass verifies the target modules and examples, **not package-wide
importability**. A normal package-level regression run remains necessary after
the aggregate import migration is complete.

## Documentation coverage and migration

There are 50 criterion pages, one [overview](index.md) and this report, all
registered in MkDocs. Twelve missing generated-criterion pages were added;
the other pages were updated to current `ParameterValues`, fitting and tail
contracts. Each page has a scientific link, with background-only and unverified
attribution explicitly labelled. Broken references to absent root-level
exponential audits were removed from these pages.

The current API uses separate KS D+/D− classes, `binning` rather than `bins`,
and Pearson code `CHI2_PEARSON`. Old critical values must match the statistic,
hypothesis, sample size, selected tail and settings; a label alone is not enough.
The Monte Carlo infinity fix additionally changes behavior for previously
rejected sparse-cell simulations. Historical scientific alternatives were
not silently replaced during this audit.

## Scientific bibliography

The formula tables above link each criterion to the relevant article(s).
The following list collects their full titles and DOI or article links.
A citation is not by itself confirmation of every implementation convention:
review articles, related models and background-only sources are explicitly
qualified in the tables and in the findings above. In particular, no primary
source has been identified for the six range/10 graph tests. WE has a published
name; its normalization correction is documented above.

1. [Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337).
2. [Lilliefors (1969), On the Kolmogorov–Smirnov test for the exponential distribution with mean unknown](https://doi.org/10.1080/01621459.1969.10500983).
3. [Anderson and Darling (1952), Asymptotic theory of certain goodness of fit criteria based on stochastic processes](https://doi.org/10.1214/aoms/1177729437).
4. [Liao and Shimokawa (1999), A new goodness-of-fit test for Type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965).
5. [Watson (1961), Goodness-of-fit tests on a circle](https://doi.org/10.1093/biomet/48.1-2.109).
6. [Kuiper (1960), Tests concerning random points on a circle](https://doi.org/10.1016/S1385-7258(60).
7. [Lemeshko et al. (2014), Using the Kuiper and Watson tests of goodness-of-fit for composite hypotheses](https://doi.org/10.3103/S875669901401004X).
8. [Cressie and Read (1984), Multinomial goodness-of-fit tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x).
9. [Epps and Pulley (1986), A test of exponentiality vs. monotone-hazard alternatives derived from the empirical characteristic function](https://doi.org/10.1111/j.2517-6161.1986.tb01403.x).
10. [Volkova and Nikitin (2013/2015), Exponentiality tests based on Ahsanullah’s characterization and their efficiencies](https://www.mathnet.ru/eng/znsl5652).
11. [Mimoto and Zitikis (2008), The Atkinson index, the Moran statistic, and testing exponentiality](https://doi.org/10.14490/jjss.38.187).
12. [Henze and Meintanis (2005), Recent and classical tests for exponentiality: a partial review with comparisons](https://doi.org/10.1007/s001840400322).
13. [Deshpande (1983), A class of tests for exponentiality against increasing failure rate average alternatives](https://doi.org/10.1093/biomet/70.2.514).
14. [Epstein (1960), Tests for the validity of the assumption that the underlying distribution of life is exponential: Part II](https://doi.org/10.1080/00401706.1960.10489892).
15. [Gail and Gastwirth (1978), A scale-free goodness-of-fit test for the exponential distribution based on the Gini statistic](https://doi.org/10.1111/j.2517-6161.1978.tb01048.x).
16. [Harris (1976), A note on testing for exponentiality](https://doi.org/10.1002/nav.3800230116).
17. [Hegazy and Green (1975), Some new goodness-of-fit tests using order statistics](https://doi.org/10.2307/2347090).
18. [Hollander and Proschan (1972), Testing whether new is better than used](https://doi.org/10.1214/aoms/1177692466).
19. [Kimber (1985), Tests for the exponential, Weibull and Gumbel distributions based on the stabilized probability plot](https://doi.org/10.1093/biomet/72.3.661).
20. [Kochar (1985), Testing exponentiality against monotone failure rate average](https://doi.org/10.1080/03610928508828919).
21. [Gail and Gastwirth (1978), A scale-free goodness-of-fit test for the exponential distribution based on the Lorenz curve](https://doi.org/10.1080/01621459.1978.10480100).
22. [Shapiro and Wilk (1972), An analysis of variance test for the exponential distribution (complete samples)](https://doi.org/10.1080/00401706.1972.10488921).
23. [Spinelli and Stephens (1987), Tests for exponentiality when origin and scale parameters are unknown, §3](https://www.stat.cmu.edu/technometrics/80-89/VOL-29-04/v2904471.pdf).
24. [Volkova (2009/2010), On asymptotic efficiency of exponentiality tests based on Rossberg’s characterization](https://doi.org/10.1007/s10958-010-9934-9).
25. [Arendarczyk, Kozubowski and Panorska (2022), The Greenwood statistic, stochastic dominance, clustering and heavy tails](https://doi.org/10.1111/sjos.12520).
26. [Wong and Wong (1979), An extremal quotient test for exponential distributions](https://doi.org/10.1007/BF01893465).
27. [Baringhaus and Henze (1991), A class of consistent tests for exponentiality based on the empirical Laplace transform](https://www.ism.ac.jp/editsec/aism/pdf/043_3_0551.pdf).
28. [Ebner (2023), The test of exponentiality based on the mean residual life function revisited](https://arxiv.org/abs/2209.05153).
29. [Baringhaus and Henze (2008)](https://doi.org/10.1016/j.spl.2007.09.060).
30. [Dettmann and Georgiou (2016), Random geometric graphs with general connection functions](https://arxiv.org/abs/1411.3617).

31. [Ascher (1990), A survey of tests for exponentiality, 19(5), 1811–1825](https://doi.org/10.1080/03610929008830292).
32. [On sensitivity of exponentiality tests to data rounding: a Monte Carlo simulation study](https://doi.org/10.1080/03610918.2021.2009868).
