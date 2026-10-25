# Lockhart–O’Reilly–Stephens spacing Anderson–Darling statistic

`LockhartOReillyStephensWeibullGofStatistic` (`LOS`) tests the ordinary
zero-location Weibull family with unknown positive shape $k$ and scale $\eta$:

$$
F(x)=1-\exp[-(x/\eta)^k],\qquad x>0.
$$

Use the full class name in imports from `continuous.weibull` or `continuous`.
The former abbreviated name has been removed without an alias; serialized
class references must also be updated. The short code `LOS` and numerical
formula are unchanged.

## Constructor and observations

Construct with `WeibullDistributionDescriptor.DEFAULT.parse({})`.
`execute_statistic(rvs, **kwargs)` requires a finite, strictly positive,
nonconstant one-dimensional sample with $n\ge3$. It returns a float,
does not modify the input and performs no parameter fitting. Common-interface
keywords are unused. Invalid samples, logarithms indistinguishable at
floating-point precision, or nonpositive approximate expected spacings raise
`ValueError`. Only complete samples are supported.

## Formula and primary-source verification

Let $y_i=\log X_{(i)}$ and $m_i=\mathbb E[\log E_{(i)}]$ for independent
unit exponential variables $E_j$. Set

$$
h_i=\frac{y_{i+1}-y_i}{m_{i+1}-m_i},\quad
 g_i=\frac{h_i}{\sum_{l=1}^{n-1}h_l},\quad i=1,\ldots,n-1,
\qquad q=n-2,\quad z_j=\sum_{i=1}^j g_i,\quad j=1,\ldots,q.
$$

The implementation substitutes approximate $m_i$ as specified below and returns

$$
A_S^2=-q-\frac1q\sum_{j=1}^q(2j-1)
\left[\log z_j+\log(1-z_{q+1-j})\right].
$$

The construction and functional were checked directly against equations
(1)–(3), pp. 344–345 of
[Lockhart, O’Reilly and Stephens (1986, JRSS B)](https://doi.org/10.1111/j.2517-6161.1986.tb01418.x),
using the [author-hosted PDF](https://www.sfu.ca/~lockhart/Research/Papers/LockhartOReillyStephensJRSSB86.pdf).
Their complete-sample indices are $k=1$, $r=n-2$; their $r+1$ normalized
spacings correspond to our $h_i$. The terminal cumulative value 1 is excluded.
The paper uses exact expected order statistics, whereas this implementation
uses an explicit approximation. There is no additional finite-sample multiplier.

### Expected-order approximation

For $Q(p)=\log[-\log(1-p)]$, the exact expectation is

$$
m_i=\frac{1}{B(i,n+1-i)}\int_0^1 Q(u)u^{i-1}(1-u)^{n-i}\,du.
$$

`GoFNS(1, n, n)` instead uses $p_i=i/(n+1)$, $q_i=1-p_i$ and

$$
\widetilde m_i=Q(p_i)+\frac{p_iq_i}{2(n+2)}Q''(p_i)
+\frac{p_iq_i}{(n+2)^2}
\left[\frac{q_i-p_i}{3}Q^{(3)}(p_i)
+\frac{p_iq_i}{8}Q^{(4)}(p_i)\right].
$$

This truncates at the fourth derivative and also approximates higher beta
central moments: the third-moment denominator $(n+2)(n+3)$ is replaced by
$(n+2)^2$, and the fourth moment by $3p_i^2q_i^2/(n+2)^2$.
It is not exact quadrature and is not asserted to reproduce the papers' tables.

## Mathematical properties and calibration

For the empirical CDF $H_q$ of $z_1,\ldots,z_q$, our algebra gives

$$
A_S^2=q\int_0^1\frac{[H_q(u)-u]^2}{u(1-u)}\,du.
$$

Thus the statistic is nonnegative in exact arithmetic and unbounded above.
Ties in the first or last ordered log gap give $z_1=0$ or $z_q=1$ and positive
infinity. Interior ties may give repeated interior $z_j$ and a finite result.
Constant samples are rejected before this calculation. For $n=3$,
$A_S^2=-1-\log[z_1(1-z_1)]$ is nondegenerate.

Under $X\mapsto aX^b$, $a,b>0$, every $h_i$ is multiplied by $b$; the $z_j$
and the statistic are unchanged. Hence shape and scale are eliminated without
fitting. The implementation computes $1-z_j$ by reverse cumulative sums of
$g_i$, avoiding subtraction from a value rounded to 1.

Large values reject (`RightAlternative`). Calibrate the upper $1-\alpha$
quantile by simulating complete Weibull samples of the observed size with
shape = scale = 1 and evaluating this same implementation.
As the authors explain on p. 345, the transformed values are **not uniform
order statistics** for a general parent distribution, including log-Weibull.
Ordinary uniform Anderson–Darling critical values therefore do not apply.
No chi-square limit is asserted. Continuous-null simulation does not model
rounding-induced ties.

The [companion Weibull article](https://doi.org/10.1002/nav.3800330307)
studies critical values and power, including censoring. Its full text was not
used for direct formula verification; this API supports complete samples only.
See [mathematical verification](mathematical-validation.md#lockhartoreillystephens-follow-up-verification).

## Example

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    LockhartOReillyStephensWeibullGofStatistic,
)

test = LockhartOReillyStephensWeibullGofStatistic(Weibull.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

- Lockhart, R. A., O’Reilly, F. J., and Stephens, M. A. (1986).
  *Tests of Fit Based on Normalized Spacings*. Journal of the Royal Statistical
  Society, Series B, 48(3), 344–352.
  [DOI](https://doi.org/10.1111/j.2517-6161.1986.tb01418.x).
- Lockhart, R. A., O’Reilly, F. J., and Stephens, M. A. (1986).
  *Tests for the extreme value and Weibull distributions based on normalized
  spacings*. Naval Research Logistics Quarterly, 33(3), 413–421.
  [DOI](https://doi.org/10.1002/nav.3800330307).
