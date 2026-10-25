# Shapiro–Brain W test for the Weibull family

`ShapiroBrainWeibullGofStatistic` uses code `SB_WEIBULL_GOODNESS_OF_FIT`.
It tests the ordinary Weibull family with zero location and unknown positive
shape and scale. It uses ordered log observations, without an MLE fit.

## Usage

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    ShapiroBrainWeibullGofStatistic,
)

criterion = ShapiroBrainWeibullGofStatistic(Weibull.DEFAULT.parse({}))
statistic = criterion.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(statistic)
```

`parameters` must declare both parameters unknown; fixed or partially fixed
parameters are unsupported. `rvs` must contain at least two finite, strictly
positive, nonconstant observations in one dimension. Ties are accepted.
The sample is sorted internally without modifying the caller's array.

## Definition

Let $y_i=\log X_{(i)}-\overline{\log X}$, in ascending order. Define

$$
w_i=\log\frac{n+1}{n-i+1},\quad
v_i=w_i(1+\log w_i)-1,\qquad i=1,\ldots,n-1,
$$

$$
w_n=n-\sum_{i=1}^{n-1}w_i,\qquad
v_n=0.4228n-\sum_{i=1}^{n-1}v_i,
$$

$$
b=\frac1n\sum_{i=1}^n(0.6079v_i-0.2570w_i)y_i,\qquad
W=\frac{nb^2}{\sum_{i=1}^n y_i^2}.
$$

The coefficients and normalization agree with the Shapiro–Brain formula
reproduced in [Shell (2003), equation (9)](https://ipnpr.jpl.nasa.gov/progress_report/42-155/155B.pdf).
The original article is [Shapiro and Brain (1987)](https://doi.org/10.1080/03610918708812586).

## Centering and mathematical checks

Centering is an explicit adjustment to the formula with rounded coefficients.
Our algebraic check gives

$$
\delta=0.6079\cdot0.4228-0.2570=0.00002012,\qquad
\sum_i(0.6079v_i-0.2570w_i)=n\delta.
$$

Consequently, evaluating the same weights on uncentered logs gives
$b_{\rm raw}=b+\delta\overline{\log X}$. Without the subtraction, changing the
units of observation changes the statistic. This implementation centers first
and therefore has $W(cX^d)=W(X)$ for $c,d>0$, up to floating-point error:
centered logs and $b$ scale by $d$, so numerator and denominator scale by $d^2$.
The denominator is positive for accepted nonconstant log samples and $W\ge0$.
This ratio is not normalized as a squared correlation, so a bound of one is not
assumed. At $n=2$ it is constant over nonconstant samples and has no discriminatory
power, although the numerical formula is defined.

Shell's Table 1 sample reproduces the report's rounded $W=0.4930$ with the
uncentered equation: our calculation gives $0.4930407980$. The centered library
variant gives $0.4929256593$. Tests distinguish these values rather than claiming
bit-for-bit equality to the uncentered expression.

## Rejection region and calibration

Both small and large values reject, following the two-quantile comparison in
[Shell (2003), Section VI](https://ipnpr.jpl.nasa.gov/progress_report/42-155/155B.pdf).
`alternative()` returns `TwoSidedAlternative`; `execute_statistic` returns only
the statistic. This corrects the former left-tail-only declaration.

Calibrate the centered variant at the same sample size. Log-location-scale
invariance permits simulation at Weibull shape = scale = 1. Every replicate
recomputes the ordered-log statistic; there is no parameter-fitting stage.
Use matching two-tail critical values, not fully specified KS tables or a
chi-squared approximation. Existing unversioned storage calibrations remain
blocked; the tail correction also changes interpretation of old decisions.

## References

1. Shapiro, S. S., and Brain, C. W. (1987).
   [W-test for the Weibull distribution](https://doi.org/10.1080/03610918708812586).
   *Communications in Statistics — Simulation and Computation*, 16(1), 209–219.
2. Shell, J. (2003).
   [Parametric Evaluation of Lifetime Data](https://ipnpr.jpl.nasa.gov/progress_report/42-155/155B.pdf).
   *IPN Progress Report 42-155*, §III.B, equation (9), and §VI, Table 1.

See the [catalog](index.md) and [mathematical verification](mathematical-validation.md).
