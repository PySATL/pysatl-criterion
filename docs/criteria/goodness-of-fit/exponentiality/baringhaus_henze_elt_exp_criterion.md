# Baringhaus–Henze empirical Laplace transform exponential goodness-of-fit test

## Hypothesis and statistic

Category: ELT / differential characterization. Parameters: Unknown rate; a fixed before seeing data, default 1; implemented 0<a<=100, theory a>0.
The origin is known to be zero. The null is Exp(rate=lambda), lambda>0;
for specified parameters it is Exp(lambda0). Scale theta=1/lambda.
Write n for sample size, x_(i) for ordered observations,
y_i=x_i/mean(x), and u_i=1-exp(-lambda*x_(i)). In fitted EDF mode
lambda=1/mean(x); otherwise it is the specified rate.

$$
T_{n,a}=n\int_0^\infty[(1+t)L_n\prime(t)+L_n(t)]^2 e^{-at}\,dt,\quad L_n(t)=n^{-1}\sum_i e^{-t y_i}
$$

Omnibus: Yes, under the conditions below.
Universally consistent: Against nonexponential alternatives with positive finite mean. Rejection: Right tail.

## Calibration and limitations

Weighted infinite sum of independent chi-square(1) variables, weights depend on a; not a single chi-square. Table 2 gives c=.303 for n=20, a=1, alpha=.05.

Finite-sample critical values and p-values are obtained by
`monte_carlo_exponential(statistic, sample, random_state=..., n_resamples=...)`.
Every simulated sample repeats all fitting and transformations, with the same n
and tuning settings. Known rate uses scale=1/rate; fitted mode can use Exp(1)
by scale invariance. Simple-hypothesis EDF tables are not used after fitting.
The p-value uses corrected ranks (b+1)/(B+1); two-tail tests double the smaller
corrected tail. The decision uses p<=alpha. Empirical critical quantiles are
summaries and can differ from the rank decision at finite B. No exact finite-n
closed-form CDF is implemented for this criterion.

Uses the analytically integrated pair kernel in the primary paper, O(n²) time and O(n) auxiliary memory. Extremely small a with zero data may exceed floating-point range and raise ValueError. Nondefault a bypasses stored calibration.

Observations must be real, finite, nonnegative and one-dimensional. Masked
observations, negative values, NaN, infinity and empty samples are rejected.
Fitted mode requires n>=2 and positive total; known-rate EDF accepts n>=1,
including all zeros. Positive constant samples are allowed. Scale normalization
uses a power of two to prevent overflow; unrepresentable dynamic ranges raise
ValueError. Independent continuous uncensored data are assumed; rounding,
truncation and censoring require their own calibration.

## Implementation and verification

[Implementation](https://github.com/PySATL/pysatl-criterion/blob/main/src/pysatl_criterion/statistics/goodness_of_fit/exponent.py):
`BaringhausHenzeELTExponentialityGofStatistic`, short code `BH_ELT`.
EDF adapters reuse common kernels; no new distribution-specific EDF formula is introduced.
[Unit tests](https://github.com/PySATL/pysatl-criterion/blob/main/tests/distributions/test_exponential_verification.py)
include independent integral/analytic references and numerical boundaries.
[Statistical validation](https://github.com/PySATL/pysatl-criterion/blob/main/tests/statistical/test_exponential_validation.py)
and [recorded results](../../../exponential-monte-carlo.json) use disjoint
calibration and validation samples and fixed seeds.
See [audit](../../../exponential-verification.md) for the provenance limits.

## References

Baringhaus, L. and Henze, N. (1991). A class of consistent tests for exponentiality based on the empirical Laplace transform. Ann. Inst. Statist. Math. 43, 551–564. DOI: https://doi.org/10.1007/BF00053372. [Primary full text](https://www.ism.ac.jp/editsec/aism/pdf/043_3_0551.pdf), pp.552–553, Theorem 2.1, Table 2.
