# Coles stabilized probability-plot criterion

`ColesWeibullGofStatistic` (short code `SPP`) uses **Blom estimates**, replacing
the previous MLE implementation. It tests the complete, uncensored two-parameter
Weibull family with zero location and unknown positive shape $k$ and scale $\eta$:

$$F(x;k,\eta)=1-\exp[-(x/\eta)^k],\qquad x>0.$$

Construct with `WeibullDistributionDescriptor.DEFAULT.parse({})`.

## Statistic and rejection region

For sorted $y_i=\log X_{(i)}$, let $z_i=(y_i-\widehat\mu_B)/\widehat\sigma_B$
and $u_i=1-\exp[-\exp(z_i)]$. With $h(p)=2\arcsin\sqrt p/\pi$, return

$$T=\max_{1\le i\le n}\left|h\!\left(\frac{i-1/2}{n}\right)-h(u_i)\right|.$$

The transform and maximum distance match
[Krit (2014), (3.14)–(3.15)](https://theses.hal.science/tel-01126901/document).
[Coles (1989)](https://doi.org/10.1093/biomet/76.3.593) studies Michael's maximum
distance using Blom estimates and also a separate correlation statistic.
This implementation returns the unscaled maximum distance, not that correlation
or a normalizing transformation of it. Large distances reject: the critical
region is the **right tail**, despite the absolute value measuring discrepancies
in both directions. There is no additional $\sqrt n$ multiplier.

## Blom estimates and approximation convention

The linear estimator follows
[`EWGoF::BLOMEst`](https://github.com/cran/EWGoF/blob/master/R/BLOMEst.R).
Let $p_i=i/(n+1)$, $\theta_i=-(1-p_i)\log(1-p_i)$, and let $m_i$ approximate
the expected $i$th order statistic of the standard minimum-Gumbel distribution.
Pad $\theta$, $\theta m$, and $\theta y$ with zero endpoints. For $j=0,\ldots,n$,
form

$$
A_j=(\theta_j-\theta_{j+1},\ \theta_jm_j-\theta_{j+1}m_{j+1}),\qquad
b_j=\theta_jy_j-\theta_{j+1}y_{j+1}.
$$

Then $(\widehat\mu_B,\widehat\sigma_B)$ minimizes
$\|A(\mu,\sigma)^T-b\|^2$. This is algebraically equivalent to the explicit
linear weights in `BLOMEst`; the implementation uses a least-squares solver
instead of explicitly inverting $A^TA$. The corresponding Weibull estimates
would be $\widehat\eta=\exp(\widehat\mu_B)$ and $\widehat k=1/\widehat\sigma_B$;
only standardized logs are needed, avoiding exponentiation of the location.

The library reuses its `GoFNS(1,n,n)` approximation. For
$Q(p)=\log[-\log(1-p)]$, $q=1-p$, it uses

$$
m_i=Q(p_i)+\frac{p_iq_i}{2(n+2)}Q''(p_i)
+\frac{p_iq_i}{(n+2)^2}
\left[\frac{q_i-p_i}{3}Q^{(3)}(p_i)+\frac{p_iq_i}{8}Q^{(4)}(p_i)\right].
$$

This is a fourth-order Taylor approximation, not exact quadrature of expected
order statistics. The fractions $1/3$ and $1/8$ are evaluated as real numbers.
The [EWGoF C++ source](https://github.com/cran/EWGoF/blob/master/src/GoFNS.cpp)
uses integer division at those two coefficients, so its compiled output is not
an exact numerical oracle for this convention. Also, `EWGoF::WPP.test` itself
uses MLE in its SPP branch: only `BLOMEst` supplies our estimator reference.
The Coles abstract confirms Blom estimation; the full original article was
not directly available. Exact reproduction of its critical-value tables is
therefore not claimed.

## Mathematical properties and calibration

Since $h$ maps $[0,1]$ into $[0,1]$, $0\le T\le1$. Under
$X\mapsto aX^d$, $a,d>0$, the response becomes $b'=\log(a)A_{:,1}+d b$.
Consequently $\widehat\mu'_B=\log(a)+d\widehat\mu_B$ and
$\widehat\sigma'_B=d\widehat\sigma_B$, leaving every $z_i$ and $T$ unchanged.
The code shifts and rescales logs before solving for numerical stability;
this is the same fit in different coordinates.

For $X=\eta E^{1/k}$, $E\sim\operatorname{Exp}(1)$, this invariance eliminates
the two unknown parameters. Calibrate at shape = scale = 1, **re-estimating
with Blom on every replicate**, at the actual sample size and with this
approximation convention. Old MLE critical values must be regenerated.
The existing storage guard rejects unversioned Weibull critical values.
Neither standard KS tables, a chi-square law, nor a standard-normal null law
applies automatically. Kimber's tables use a different estimator, as noted
by Krit, and are not substituted here.

For $n=2$, any two nonconstant ordered log samples are related by a positive
affine transformation, so $T$ is constant and has no discriminatory power.
Use $n\ge3$ for testing. This does not imply that the two-observation value is zero.

## Input and numerical behavior

`execute_statistic(rvs, **kwargs)` accepts a one-dimensional finite, strictly
positive, nonconstant sample of at least two observations. It does not mutate
the sample or retain estimates. Additional common-interface keywords are unused.
Invalid samples, indistinguishable logs, nonpositive fitted log-scale or an
unrepresentable fit raise `ValueError`. Ties are accepted; when observations
are rounded, calibration must reproduce rounding. Censoring and unknown
location are not supported. CDF saturation at zero or one gives a finite
endpoint value of $h$.

## Example

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import ColesWeibullGofStatistic

test = ColesWeibullGofStatistic(Weibull.DEFAULT.parse({}))
print(test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2]))
```

See [the verification report](mathematical-validation.md#coles-follow-up-verification)
for numerical tests and source qualifications.

## References

- Coles, S. G. (1989). *On goodness-of-fit tests for the two-parameter Weibull
  distribution derived from the stabilized probability plot*. Biometrika,
  76(3), 593–598. [DOI](https://doi.org/10.1093/biomet/76.3.593).
- Michael, J. R. (1983). *The stabilized probability plot*. Biometrika, 70(1),
  11–17. [DOI](https://doi.org/10.1093/biomet/70.1.11).
- Krit, M. (2014). *Goodness-of-fit tests in reliability: Weibull distribution
  and imperfect maintenance models*. PhD thesis, §3.2.1, (3.14)–(3.15), p. 48.
  [Full text](https://theses.hal.science/tel-01126901/document).
- Krit, M. *EWGoF*: [BLOMEst source](https://github.com/cran/EWGoF/blob/master/R/BLOMEst.R)
  and [GoFNS source](https://github.com/cran/EWGoF/blob/master/src/GoFNS.cpp).
- Kimber, A. C. (1985). *Tests for the exponential, Weibull and Gumbel
  distributions based on the stabilized probability plot*. Biometrika, 72(3),
  661–663. [DOI](https://doi.org/10.1093/biomet/72.3.661).
