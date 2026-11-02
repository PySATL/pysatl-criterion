# Tiku–Singh normalized log-spacing statistic

`TikuSinghWeibullGofStatistic` (`TS`) tests the ordinary two-parameter Weibull
family with zero location and unknown shape and scale:

$$
F(x)=1-\exp[-(x/\eta)^k],\qquad x>0,\quad k,\eta>0.
$$

## Construction and sample requirements

Pass `WeibullDistributionDescriptor.DEFAULT.parse({})` to the constructor.
`execute_statistic(rvs, **kwargs)` returns a scalar statistic; common-interface
keywords are unused. The sample must be one-dimensional, finite, strictly
positive, nonconstant and have $n\ge3$ observations. Invalid samples,
logarithms indistinguishable at floating-point precision, or nonpositive
approximate expected spacings raise `ValueError`.
The input is not modified. No parameters are fitted. Only complete samples
are supported. Ties are accepted computationally; continuous-null simulation
does not account for a measurement-rounding mechanism.

## Statistic

Let $y_i=\log X_{(i)}$, $i=1,\ldots,n$. For independent unit exponential
variables $E_j$, let $m_i=\mathbb E[\log E_{(i)}]$. Define

$$
h_i=\frac{y_{i+1}-y_i}{m_{i+1}-m_i},\quad
 g_i=\frac{h_i}{\sum_{j=1}^{n-1}h_j},\qquad i=1,\ldots,n-1,
$$

$$
T=\frac{2}{n-2}\sum_{i=1}^{n-2}(n-1-i)g_i.
$$

The implementation replaces $m_i$ by the explicit approximation below.
This functional is $Z^*$ in [Sürücü (2008), (3.4)–(3.5)](https://doi.org/10.1016/j.camwa.2008.03.010),
applied to log observations. Its Weibull attribution is
[Tiku and Singh (1981)](https://doi.org/10.1080/03610928108828082), building on
[Tiku (1980)](https://doi.org/10.1111/j.1467-842X.1980.tb01175.x).

[Krit (2014), (30)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf)
reproduces TS, but its preceding definition of $E_i$ uses observations
$i,i+1$ while (30) sums over $i=2,\ldots,n$. These indices are inconsistent.
Interpreting $E_i=h_{i-1}$ in (30) gives the formula above. The original
1981 full text was not checked directly; exact agreement with its tables
or expectation approximations is not asserted.

### Expected-order approximation

For $U_{(i)}\sim\mathrm{Beta}(i,n+1-i)$ and
$Q(p)=\log[-\log(1-p)]$, the exact mean is

$$
m_i=\frac{1}{B(i,n+1-i)}\int_0^1
Q(u)u^{i-1}(1-u)^{n-i}\,du.
$$

`GoFNS(1, n, n)` instead uses $p_i=i/(n+1)$, $q_i=1-p_i$ and

$$
\widetilde m_i=Q(p_i)+\frac{p_iq_i}{2(n+2)}Q''(p_i)
+\frac{p_iq_i}{(n+2)^2}
\left[\frac{q_i-p_i}{3}Q^{(3)}(p_i)
+\frac{p_iq_i}{8}Q^{(4)}(p_i)\right].
$$

This is a fourth-derivative Taylor expansion with approximate higher beta
central moments, not an exact fourth-order moment calculation. In particular,
the exact third central moment is $2p_iq_i(q_i-p_i)/[(n+2)(n+3)]$;
the code replaces $(n+2)(n+3)$ by $(n+2)^2$ and uses
$3p_i^2q_i^2/(n+2)^2$ for the fourth central moment.
This approximation is part of the implemented definition and can affect
finite-sample critical values, particularly through extreme order statistics.

## Rejection and calibration

Both small and large values reject (`TwoSidedAlternative`). Use sample-size-specific
Monte Carlo quantiles of this same implementation, for example lower
$\alpha/2$ and upper $1-\alpha/2$ quantiles for an equal-tail level-$\alpha$ test.
Do not assume that these quantiles are symmetric around 1.

Under $X\mapsto aX^b$ with $a,b>0$, log spacings are multiplied by $b$,
so $g_i$ and $T$ are unchanged. Therefore simulation can use shape = scale = 1
without fitting parameters in any replicate.

Since $g_i\ge0$ and $\sum g_i=1$, $T$ is a weighted average of numbers in
$[0,2]$. Thus $0\le T\le2$ in exact arithmetic, and $0<T<2$ for distinct
observations. With $z_j=\sum_{i=1}^j g_i$,
$T=2\sum_{j=1}^{n-2}z_j/(n-2)$. These $z_j$ are not assumed to be independent
uniform variables under a log-Weibull null. In particular, the exact
uniform-mean law for ordinary exponential spacings does not supply a
calibration for this generalized log-spacing statistic.

Historical critical values need not match the approximation used here.
Generate fresh calibration with `execute_statistic`; no chi-square limit or
censored-data calibration is supplied. See the
[mathematical verification](mathematical-validation.md#tikusingh-follow-up-verification).

## Example

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    TikuSinghWeibullGofStatistic,
)

test = TikuSinghWeibullGofStatistic(Weibull.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

- Tiku, M. L. (1980). *Goodness of fit statistics based on the spacings of
  complete or censored samples*. Australian Journal of Statistics, 22(3),
  260–275. [DOI](https://doi.org/10.1111/j.1467-842X.1980.tb01175.x).
- Tiku, M. L., and Singh, M. (1981). *Testing the two parameter Weibull
  distribution*. Communications in Statistics — Theory and Methods, 10(9),
  907–918. [DOI](https://doi.org/10.1080/03610928108828082).
- Sürücü, B. (2008). *A power comparison and simulation study of goodness-of-fit
  tests*. Computers & Mathematics with Applications, 56(6), 1617–1625.
  [DOI](https://doi.org/10.1016/j.camwa.2008.03.010).
- Krit, M. (2014). *Goodness-of-fit tests for the Weibull distribution based on
  the Laplace transform*. Journal de la Société Française de Statistique,
  155(3), 135–151. [Article](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf).
