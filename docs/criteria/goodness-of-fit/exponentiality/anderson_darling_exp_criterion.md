# Anderson–Darling exponential goodness-of-fit test

## Hypothesis and statistic

Category: EDF. Parameters: Both known and fitted rate.
The origin is known to be zero. The null is Exp(rate=lambda), lambda>0;
for specified parameters it is Exp(lambda0). Scale theta=1/lambda.
Write n for sample size, x_(i) for ordered observations,
y_i=x_i/mean(x), and u_i=1-exp(-lambda*x_(i)). In fitted EDF mode
lambda=1/mean(x); otherwise it is the specified rate.

$$
A^2=-n-\frac1n\sum_{i=1}^n(2i-1)[\log u_i+\log(1-u_{n+1-i})]
$$

Omnibus: Yes, under the conditions below.
Universally consistent: Yes. Rejection: Right tail.

## Calibration and limitations

For fixed CDF, a Brownian-bridge weighted squared integral; fitting changes the limit. No normal or single chi-square approximation is supplied.

Finite-sample critical values and p-values are obtained by
`monte_carlo_exponential(statistic, sample, random_state=..., n_resamples=...)`.
Every simulated sample repeats all fitting and transformations, with the same n
and tuning settings. Known rate uses scale=1/rate; fitted mode can use Exp(1)
by scale invariance. Simple-hypothesis EDF tables are not used after fitting.
The p-value uses corrected ranks (b+1)/(B+1); two-tail tests double the smaller
corrected tail. The decision uses p<=alpha. Empirical critical quantiles are
summaries and can differ from the rank decision at finite B. No exact finite-n
closed-form CDF is implemented for this criterion.

A zero observation gives +infinity. Log survival is evaluated directly as -lambda*x, avoiding CDF saturation.

Observations must be real, finite, nonnegative and one-dimensional. Masked
observations, negative values, NaN, infinity and empty samples are rejected.
Fitted mode requires n>=2 and positive total; known-rate EDF accepts n>=1,
including all zeros. Positive constant samples are allowed. Scale normalization
uses a power of two to prevent overflow; unrepresentable dynamic ranges raise
ValueError. Independent continuous uncensored data are assumed; rounding,
truncation and censoring require their own calibration.

## Implementation and verification

[Implementation](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/exponent.py):
`AndersonDarlingExponentialityGofStatistic`, short code `AD`.
EDF adapters reuse common kernels; no new distribution-specific EDF formula is introduced.
[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/distributions/test_exponential_verification.py)
include independent integral/analytic references and numerical boundaries.
[Statistical validation](https://github.com/PySATL/pysatl-criterion/blob/main/tests/statistical/test_exponential_validation.py)
and [recorded results](../../../exponential-monte-carlo.json) use disjoint
calibration and validation samples and fixed seeds.
See [audit](../../../exponential-verification.md) for the provenance limits.

## References

Anderson, T. W. and Darling, D. A. (1952). Asymptotic theory of certain goodness of fit criteria. DOI: https://doi.org/10.1214/aoms/1177729437
