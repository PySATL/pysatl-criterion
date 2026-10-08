# Pearson chi-square exponential goodness-of-fit test

## Hypothesis and statistic

Category: Binned EDF. Parameters: Both known and fitted rate; bins=k>=2 integer, default 5.
The origin is known to be zero. The null is Exp(rate=lambda), lambda>0;
for specified parameters it is Exp(lambda0). Scale theta=1/lambda.
Write n for sample size, x_(i) for ordered observations,
y_i=x_i/mean(x), and u_i=1-exp(-lambda*x_(i)). In fitted EDF mode
lambda=1/mean(x); otherwise it is the specified rate.

$$
X^2=\sum_{j=1}^k\frac{(N_j-n/k)^2}{n/k}
$$

Omnibus: Binned test; limited by fixed bins.
Universally consistent: No with fixed k. Rejection: Right tail.

## Calibration and limitations

Known rate: asymptotic chi-square(k-1). With ungrouped MLE and fitted bin boundaries, chi-square(k-2) is NOT assumed.

Finite-sample critical values and p-values are obtained by
`monte_carlo_exponential(statistic, sample, random_state=..., n_resamples=...)`.
Every simulated sample repeats all fitting and transformations, with the same n
and tuning settings. Known rate uses scale=1/rate; fitted mode can use Exp(1)
by scale invariance. Simple-hypothesis EDF tables are not used after fitting.
The p-value uses corrected ranks (b+1)/(B+1); two-tail tests double the smaller
corrected tail. The decision uses p<=alpha. Empirical critical quantiles are
summaries and can differ from the rank decision at finite B. No exact finite-n
closed-form CDF is implemented for this criterion.

Bins have boundaries -log(1-j/k)/lambda, j=1,...,k-1, plus 0 and infinity. Boundary ties enter the right bin. Sparse counts need simulation; alternatives with identical bin probabilities cannot be detected.

Observations must be real, finite, nonnegative and one-dimensional. Masked
observations, negative values, NaN, infinity and empty samples are rejected.
Fitted mode requires n>=2 and positive total; known-rate EDF accepts n>=1,
including all zeros. Positive constant samples are allowed. Scale normalization
uses a power of two to prevent overflow; unrepresentable dynamic ranges raise
ValueError. Independent continuous uncensored data are assumed; rounding,
truncation and censoring require their own calibration.

## Implementation and verification

[Implementation](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/exponent.py):
`Chi2PearsonExponentialityGofStatistic`, short code `CHI2`.
EDF adapters reuse common kernels; no new distribution-specific EDF formula is introduced.
[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/distributions/test_exponential_verification.py)
include independent integral/analytic references and numerical boundaries.
[Statistical validation](https://github.com/PySATL/pysatl-criterion/blob/main/tests/statistical/test_exponential_validation.py)
and [recorded results](../../../exponential-monte-carlo.json) use disjoint
calibration and validation samples and fixed seeds.
See [audit](../../../exponential-verification.md) for the provenance limits.

## References

Pearson, K. (1900). On the criterion that a given system of deviations ... DOI: https://doi.org/10.1080/14786440009463897
