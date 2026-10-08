# Greenwood exponential goodness-of-fit test

## Hypothesis and statistic

Category: Normalized observations / uniform-spacings distribution. Parameters: Unknown rate.
The origin is known to be zero. The null is Exp(rate=lambda), lambda>0;
for specified parameters it is Exp(lambda0). Scale theta=1/lambda.
Write n for sample size, x_(i) for ordered observations,
y_i=x_i/mean(x), and u_i=1-exp(-lambda*x_(i)). In fitted EDF mode
lambda=1/mean(x); otherwise it is the specified rate.

$$
G=\frac{\sum_i x_i^2}{(\sum_i x_i)^2}
$$

Omnibus: No: a moment contrast, not universally consistent.
Universally consistent: No. Rejection: Both tails.

## Calibration and limitations

Exact moments: E(G)=2/(n+1), Var(G)=4(n-1)/((n+1)^2(n+2)(n+3)). sqrt(n)*(n*G-2)/2 tends to N(0,1), slowly.

Finite-sample critical values and p-values are obtained by
`monte_carlo_exponential(statistic, sample, random_state=..., n_resamples=...)`.
Every simulated sample repeats all fitting and transformations, with the same n
and tuning settings. Known rate uses scale=1/rate; fitted mode can use Exp(1)
by scale invariance. Simple-hypothesis EDF tables are not used after fitting.
The p-value uses corrected ranks (b+1)/(B+1); two-tail tests double the smaller
corrected tail. The decision uses p<=alpha. Empirical critical quantiles are
summaries and can differ from the rank decision at finite B. No exact finite-n
closed-form CDF is implemented for this criterion.

Use original observations, not ordered exponential spacings. Their normalized shares are Dirichlet(1,...,1). Existing WE is an affine transform: WE=(n-1)*(n*G-1)/n². It is not an independent statistical family. Both tails are necessary.

Observations must be real, finite, nonnegative and one-dimensional. Masked
observations, negative values, NaN, infinity and empty samples are rejected.
Fitted mode requires n>=2 and positive total; known-rate EDF accepts n>=1,
including all zeros. Positive constant samples are allowed. Scale normalization
uses a power of two to prevent overflow; unrepresentable dynamic ranges raise
ValueError. Independent continuous uncensored data are assumed; rounding,
truncation and censoring require their own calibration.

## Implementation and verification

[Implementation](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/exponent.py):
`GreenwoodExponentialityGofStatistic`, short code `GREENWOOD`.
EDF adapters reuse common kernels; no new distribution-specific EDF formula is introduced.
[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/distributions/test_exponential_verification.py)
include independent integral/analytic references and numerical boundaries.
[Statistical validation](https://github.com/PySATL/pysatl-criterion/blob/main/tests/statistical/test_exponential_validation.py)
and [recorded results](../../../exponential-monte-carlo.json) use disjoint
calibration and validation samples and fixed seeds.
See [audit](../../../exponential-verification.md) for the provenance limits.

## References

Greenwood, M. (1946). The Statistical Study of Infectious Diseases. HMSO (no DOI). Moran, P. A. P. (1947). The random division of an interval. DOI: https://doi.org/10.2307/2983572. Accessible formula and CV identity: Arendarczyk, Kozubowski and Panorska (2023), (1.4)-(1.5), https://doi.org/10.57805/revstat.v21i3.357
