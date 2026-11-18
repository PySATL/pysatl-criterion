# Fitted Laplace: definitions and conventions

All statistics on this page test the composite family Laplace(t, s), s > 0.
Write n for sample size, μ̂ for the sample median (average of the two middle
observations for even n), b̂ = mean|x−μ̂|, and z_i=(x_i−μ̂)/b̂.
Let x̄ be the sample mean and s_n²=mean(x−x̄)², with divisor n throughout.
Order-statistic subscripts outside 1,…,n are clamped to the nearest endpoint.
The implementation works on standardized observations to avoid scale overflow.
Constant samples have no admissible fitted scale.

## Characteristic-function tests

For j=1,2,

\[
T_{n,a}^{(j)}=n\int_{-\infty}^{\infty}
\left|(1+t^2)n^{-1}\sum_i e^{it z_i}-1\right|^2w_j(t,a)\,dt,
\quad w_1=e^{-a|t|},\quad w_2=e^{-at^2}.
\]

The default is MLE standardization. With `estimator="mom"`, replace z by
(x−x̄)/(s_n/√2). Both statistics reject for large values. The closed forms in
[Meintanis (2004)](https://doi.org/10.1081/STA-120028735) are evaluated without
allocating an n×n matrix. Default a values are 2 and 0.5, respectively.
The defining integral provides both an independent test and a cancellation fallback.

## Choi–Kim entropy tests

Choose integer 1 ≤ m < n/2 and define Δ_i=z_(i+m)−z_(i−m).

\[
H_v=\frac1n\sum_{i=1}^n\log\frac{n\Delta_i}{2m},\qquad
H_{\mathrm{vanEs}}=\frac1{n-m}\sum_{i=1}^{n-m}\log(z_{i+m}-z_i)
+\sum_{k=m}^{n}\frac1k.
\]

For Correa, let z̄_i be the average of the 2m+1 clamped observations indexed
j=i−m,…,i+m. The rank j itself is not clamped:

\[
H_c=-\frac1n\sum_i\log
\frac{\sum_{j=i-m}^{i+m}(j-i)(z_j-\bar z_i)/n}
{\sum_{j=i-m}^{i+m}(z_j-\bar z_i)^2}.
\]

CK_v=exp(H_v), CK_c=exp(H_c), CK_e=exp(H_vanEs) on standardized data;
in original units these are exp(H)/b̂. No extra 2e divisor is applied.
Small values reject. See [Choi–Kim (2006)](https://doi.org/10.1080/02331880600822473).

## AP entropy tests and corrected scales

[Alizadeh Noughabi–Park (2016)](https://doi.org/10.1080/00949655.2015.1104685)
use AP_r=log(2θ_r)+1−H_r. Their H_e is the Ebrahimi estimate, **not** van Es.
For r=v,e,a,z,

\[
H_r=\frac1n\sum_i\log\frac{n\Delta_i}{m c_{r,i}},
\]

where c_v,i=2; c_e,i=1+min(i−1,m,n−i)/m;
c_a,i=1 at i≤m or i≥n−m+1 and 2 otherwise;
c_z,i=i/m for i≤m, (n−i+1)/m for i≥n−m+1, and 2 otherwise.

For AP_y, define B=(n−1)/(n(n+1)) and the smoothed empirical CDF values

\[
F_1=B\left(\frac n{n-1}+\frac n{2n-1}\right),\quad
F_n=B\left(\frac{(n-1)^2+1}{n-1}+1+\frac{n-1}{2n-1}\right),
\]

\[
F_i=B\left(\frac{i(n-1)+1}{n-1}+
\frac{z_i-z_{i-1}}{z_{i+1}-z_{i-1}}\right),\quad 2\le i\le n-1.
\]

Set d_i=F_(i+m)−F_(i−m), C=Σd_i and
H_y=Σ(d_i/C)log(Δ_i/d_i). The implementation uses the strictly increasing
sample branch of this definition: AP_y rejects tied observations explicitly.

The corrected scale is the integral of absolute deviation of a piecewise-linear
quantile Q with knots β_i at (i−1)/n, i=1,…,n+1:

\[
\theta(\beta)=\int_0^1|Q(u)-Q(1/2)|\,du.
\]

Define ξ_i=(2m)⁻¹Σ_{k=i−m}^{i+m−1}z_k. For v use β=ξ, and for y use
β_i=2mξ_i/C. For e,a,z, interior knots m+1≤i≤n−m+1 equal ξ_i; replace tails by

\[
\beta_i=\xi_{m+1}-\sum_{k=i}^{m}\frac{z_{m+k}-z_1}{d_{L,k}}
\quad(i\le m),\qquad
\beta_i=\xi_{n-m+1}+\sum_{k=n-m+2}^{i}\frac{z_n-z_{k-m-1}}{d_{R,k}}
\quad(i\ge n-m+2).
\]

| Variant | d_L,k | d_R,k |
|---|---|---|
| e | m+k−1 | n+m−k+1 |
| a | m | m |
| z | k | n−k+2 |

The integral uses trapezoids on each side of 1/2, splitting the central interval
for odd n. `scale_estimator="mle"` substitutes θ=1 on standardized data,
providing the corresponding MLE-scale variants. Large values reject.

## Moment and smooth components

Let g_1=mean((x−x̄)/s_n)³ and g_2=mean((x−x̄)/s_n)⁴. The signed rescaled
components are

\[
\widetilde V_3=\sqrt{n/63}\,g_1,\qquad
\widetilde V_4=\sqrt{n/1188}\,(g_2-6).
\]

The moment components reported in [Batsidis et al., Section 3.3](https://www.ajs.or.at/index.php/ajs/article/download/1251/754)
use denominators 54 and 1072.8 and have asymptotic variances 7/6 and 165/149.
Rescaling gives 54·7/6=63 and 1072.8·165/149=1188. This is a reconstruction
of the unit-variance versions motivated by [Best–Rayner–Thas (2008)](https://doi.org/10.1016/j.csda.2008.05.023).
It can also be checked directly from Laplace moments: for standardized variance-one
Y, E(Y^(2k))=(2k)!/2^k. The influence functions are Y³−3Y and Y⁴−12Y²+6,
with variances 63 and 1188. Both components have two-sided rejection.

For [Gel (2010)](https://doi.org/10.1016/j.csda.2009.10.008), put
q_j=mean((x−x̄)/(√2 b̂))^j, j=3,4:

\[
Ge=nq_3^2/60+n(q_4-6)^2/1200.
\]

Large values reject. The constants are the recommended small/moderate-sample
choice, not a fitted estimate of asymptotic variances.

The signed [González-Estrada–Villaseñor (2016)](https://doi.org/10.1016/j.spl.2016.07.003)
statistic is

\[
GV=2\sqrt n\left\{\frac{s_n}{\sqrt2\,\mathrm{mean}|x-\bar x|}-1\right\}.
\]

The absolute deviations here are about the sample **mean**. Rejection is two-sided.

## Density likelihood and informational energy

For [Alizadeh Noughabi (2016)](https://doi.org/10.1007/s40304-016-0095-0),

\[
A_{rat}=\min_{1\le m<\min(n^\delta,n/2)}
\prod_{i=1}^n\frac{2m}{n\Delta_i f_{0,1}(z_i)},\qquad \delta=0.5.
\]

Only integer windows are considered. The extra n/2 bound follows the
[comparison paper, Section 3.3](https://arxiv.org/abs/2007.06154).
The calculation sums log factors and minimizes before exponentiating.
`log_scale=True` returns the log of the minimum. Large values reject.

For [Alizadeh Noughabi–Jarrahiferiz (2019)](https://www.inderscience.com/info/inarticle.php?artid=102022),

\[
AJ=\frac1n\sum_i\frac{2m}{n\{F_{0,1}(z_{i+m})-F_{0,1}(z_{i-m})\}}.
\]

Large values reject. Tail differences use exponential identities with `expm1`.

## DLO finite-sample scores

The λ=1 specialization of [Desgagné–Lafaye de Micheaux–Ouimet](https://doi.org/10.1080/02331888.2022.2144859)
is implemented with the finite-sample corrections in equations (7)–(9) of the
[comparison paper](https://arxiv.org/abs/2007.06154). Define

\[
S=\mathrm{mean}(z),\quad K=\mathrm{mean}(|z|\log|z|),\quad
K_{net}=\max(0,K-S^2/2),\quad c=1-\gamma.
\]

Use 0·log0=0 and Euler's constant γ. Then

\[
Z_S=\frac{\sqrt n S}{\sqrt{v_S}},\qquad
Z_K=\frac{\sqrt n(K_{net}^{1/4}-c^{1/4}b_K)}
{\sqrt{c^{-3/2}(\pi^2/3-3)v_K/16}}.
\]

| n | v_S | b_K | v_K |
|---|---|---|---|
| even | 1−1.856/n^1.06 | 1−0.422/n^1.01 | 1−1.950/n^0.92+39.349/n^2.3 |
| odd | 1−0.281/n^1.03 | 1−0.198/n^0.86 | 1−3.827/n^1.04 |

DLO_X=Z_S²+Z_K² rejects on the right; DLO_Z=Z_K is signed and two-sided.
Require n≥4. Normal/chi-square asymptotic approximations are not substituted for
finite-sample calibration.

## Data-driven smooth adaptation

This class specializes the general efficient-score construction and Schwarz
selection rule of [Janic–Ledwina, Sections 2–2.1](https://www.impan.pl/~ledwina/Janic-Ledwina-24-09-08.pdf)
to the Laplace MLE. It is **not** a verified implementation of the DD comparator
in Best–Rayner–Thas.

Let φ_j(u)=√(2j+1)P_j(2u−1) for Legendre polynomials P_j. The standardized
Laplace nuisance score is s(z)=(sign z, |z|−1)', with E(ss')=I₂.
Let A_j=E[φ_j(F(Z))s(Z)'] under Z∼Laplace(0,1), and
v=mean φ(F(z))−A mean s(z). For each k=1,…,d,

\[
W_k=n v_{1:k}'(I_k-A_{1:k}A_{1:k}')^{-1}v_{1:k},\qquad
\hat k=\min\arg\max_k\{W_k-k\log n\}.
\]

The statistic is W_k̂ and rejects on the right. Default d=4; integer 1≤d≤10.
Population integrals are computed at construction, not estimated from the sample.
A fixed d does not give an omnibus-consistency guarantee.

## Shared calibration and implementation settings

Constructor settings enter `configuration()`. Use identical sample size, window,
weight, estimation mode, score dimension and log/product convention when calibrating.
The built-in generator uses Laplace(0,1) for these invariant composite tests.
Every execution fits anew and leaves the hypothesis parameters unknown.

All spacing windows default to m=1. This explicit choice does not reproduce the
size-dependent power-optimized tables in the comparison paper. Ties are permitted
except in AP_y; zero required spacings yield limiting zero CK or infinite AP/AJ/A_rat.
Neither known-parameter EDF tables nor critical values for a different window apply.

[Catalog](index.md) · [Validation report](mathematical-validation.md)
