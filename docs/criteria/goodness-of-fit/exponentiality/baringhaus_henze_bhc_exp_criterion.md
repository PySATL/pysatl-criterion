# Generalized Baringhaus–Henze BHC exponential goodness-of-fit test

## Hypothesis and statistic

Category: Mean residual life characterization. Parameters: Unknown rate; fixed a, default 1; implemented 0<a<=100; negative and zero a are not supported.
The origin is known to be zero. The null is Exp(rate=lambda), lambda>0;
for specified parameters it is Exp(lambda0). Scale theta=1/lambda.
Write n for sample size, x_(i) for ordered observations,
y_i=x_i/mean(x), and u_i=1-exp(-lambda*x_(i)). In fitted EDF mode
lambda=1/mean(x); otherwise it is the specified rate.

$$
BHC_{n,a}=n\int_0^\infty\left[n^{-1}\sum_i\{\min(y_i,t)-1(y_i\leq t)\}\right]^2e^{-at}\,dt
$$

Omnibus: Yes, under the conditions below.
Universally consistent: Against nonexponential alternatives with positive finite mean (no broader claim). Rejection: Right tail.

## Calibration and limitations

Integral of a squared zero-mean Gaussian process with K(s,t)=exp(-max(s,t))-exp(-s-t), weight exp(-a*t). Mean limit 1/((a+1)(a+2)). At a=1, the ordinary CvM limit applies: c(.05) approximately .461; Ebner Table 1 approximation .463.

Finite-sample critical values and p-values are obtained by
`monte_carlo_exponential(statistic, sample, random_state=..., n_resamples=...)`.
Every simulated sample repeats all fitting and transformations, with the same n
and tuning settings. Known rate uses scale=1/rate; fitted mode can use Exp(1)
by scale invariance. Simple-hypothesis EDF tables are not used after fitting.
The p-value uses corrected ranks (b+1)/(B+1); two-tail tests double the smaller
corrected tail. The decision uses p<=alpha. Empirical critical quantiles are
summaries and can differ from the rank decision at finite B. No exact finite-n
closed-form CDF is implemented for this criterion.

The equivalent integrand mean((Y-t-1)*1{Y>t}) is integrated piecewise exactly using polynomial-exponential moments. At a=1 this is BH CvM; there is no duplicate class. Original 2000/2008 full texts were not accessible; the implementation is anchored to the explicit independent derivation in Ebner, not to an uninspected paper. The rate/scale wording in the first paragraph of that preprint is inconsistent; Y=X/mean(X) and the density resolve the convention.

Observations must be real, finite, nonnegative and one-dimensional. Masked
observations, negative values, NaN, infinity and empty samples are rejected.
Fitted mode requires n>=2 and positive total; known-rate EDF accepts n>=1,
including all zeros. Positive constant samples are allowed. Scale normalization
uses a power of two to prevent overflow; unrepresentable dynamic ranges raise
ValueError. Independent continuous uncensored data are assumed; rounding,
truncation and censoring require their own calibration.

## Implementation and verification

[Implementation](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/exponent.py):
`BaringhausHenzeBHCExponentialityGofStatistic`, short code `BH_BHC`.
EDF adapters reuse common kernels; no new distribution-specific EDF formula is introduced.
[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/distributions/test_exponential_verification.py)
include independent integral/analytic references and numerical boundaries.
[Statistical validation](https://github.com/PySATL/pysatl-criterion/blob/main/tests/statistical/test_exponential_validation.py)
and [recorded results](../../../exponential-monte-carlo.json) use disjoint
calibration and validation samples and fixed seeds.
See [audit](../../../exponential-verification.md) for the provenance limits.

## References

Baringhaus, L. and Henze, N. (2008). A new weighted integral goodness-of-fit statistic for exponentiality. DOI: https://doi.org/10.1016/j.spl.2007.09.060. Independently verified primary derivation: Ebner (2023), The test of exponentiality based on the mean residual life function revisited, [arXiv:2209.05153v2](https://publikationen.bibliothek.kit.edu/1000156446/150387001), equation (4), Remark 1.1, Appendix B. The a=1 member is the CvM-type statistic of Baringhaus–Henze (2000), DOI: https://doi.org/10.1007/BF02926105.
