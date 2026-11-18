# Mann–Scheuer–Fertig normalized log-spacing statistic

`MannScheuerFertigWeibullGofStatistic` (`MSF`) tests the zero-location Weibull
family with unknown positive shape $k$ and scale $\eta$:

$$
F(x)=1-\exp[-(x/\eta)^k],\qquad x>0.
$$

The public class name is now `MannScheuerFertigWeibullGofStatistic` in both
`continuous.weibull` and `continuous`. No alias for the former abbreviated
class name is provided. Update imports and serialized class references.
The short code `MSF` and the numerical formula are unchanged.

## Constructor and observations

Pass `WeibullDistributionDescriptor.DEFAULT.parse({})`.
`execute_statistic(rvs, **kwargs)` accepts a finite, strictly positive,
nonconstant one-dimensional sample with $n\ge3$; common-interface keywords
are unused. It returns a float, does not modify the input and performs no
parameter fitting. Invalid observations, logarithms indistinguishable at
floating-point precision, or nonpositive approximate expected spacings raise
`ValueError`. Only complete samples are supported. Ties are accepted
computationally; continuous-null simulation does not model measurement rounding.

## Formula and source correspondence

Write $y_i=\log X_{(i)}$ and $m_i=\mathbb E[\log E_{(i)}]$ for independent
unit exponential variables $E_j$. Define $r=\lfloor n/2\rfloor$ and

$$
h_i=\frac{y_{i+1}-y_i}{m_{i+1}-m_i},\qquad i=1,\ldots,n-1,
\qquad
S=\frac{\sum_{i=r+1}^{n-1}h_i}{\sum_{i=1}^{n-1}h_i}.
$$

The code substitutes approximate $m_i$ as specified below. The ratio matches
[Krit (2014), equation (29)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf)
and [Sürücü (2008), equation (4.13)](https://doi.org/10.1016/j.camwa.2008.03.010),
which attribute it to [Mann, Scheuer and Fertig (1973)](https://doi.org/10.1080/03610927308827085).
The original 1973 full text was not checked directly; agreement with its
expected-value tables is not asserted.

There are $r$ lower intervals and $n-1-r$ upper intervals. The numerator
includes the last interval. For $n=2m$ their counts are $m$ and $m-1$;
for $n=2m+1$ both counts are $m$. Thus the code's zero-based slice `g[n // 2:]`
is correct, including for even sample sizes. At $n=3$, $S=h_2/(h_1+h_2)$.

### Expected-order approximation

For $Q(p)=\log[-\log(1-p)]$, the exact mean is

$$
m_i=\frac{1}{B(i,n+1-i)}\int_0^1 Q(u)u^{i-1}(1-u)^{n-i}\,du.
$$

`GoFNS(1, n, n)` uses $p_i=i/(n+1)$, $q_i=1-p_i$ and

$$
\widetilde m_i=Q(p_i)+\frac{p_iq_i}{2(n+2)}Q''(p_i)
+\frac{p_iq_i}{(n+2)^2}
\left[\frac{q_i-p_i}{3}Q^{(3)}(p_i)
+\frac{p_iq_i}{8}Q^{(4)}(p_i)\right].
$$

Besides truncating at the fourth derivative, this expansion approximates
higher beta central moments. The exact third central moment is
$2p_iq_i(q_i-p_i)/[(n+2)(n+3)]$; the code uses $(n+2)^2$ in the denominator
and $3p_i^2q_i^2/(n+2)^2$ for the fourth central moment. This convention is
part of the implementation, not exact quadrature of expected order statistics.

## Range, rejection and calibration

Because all $h_i$ are nonnegative and their sum is positive, $0\le S\le1$;
for distinct observations $0<S<1$. Tied samples can attain either endpoint.
Under $X\mapsto aX^b$ with $a,b>0$, every $h_i$ is multiplied by $b$, so
$S$ is invariant. Consequently its null distribution depends on the sample
size and expectation approximation, but not on Weibull shape or scale.

The implementation uses `RightAlternative`: reject for large $S$, using the
upper $1-\alpha$ Monte Carlo quantile for a level-$\alpha$ test. It is an
explicitly one-sided version; alternatives producing small $S$ may have poor
power. It does not implement a two-tail rule.
Simulate complete Weibull samples of the observed size with shape = scale = 1,
calling this same implementation in each replicate; no fitting is required.

An exact beta law would require additional assumptions on joint spacing
behaviour, not merely their expected values. Neither exact beta/F calibration
nor a chi-square limit is asserted here. Historical critical values need not
match this expectation approximation. See
[mathematical verification](mathematical-validation.md#mannscheuerfertig-follow-up-verification).

## Example

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    MannScheuerFertigWeibullGofStatistic,
)

test = MannScheuerFertigWeibullGofStatistic(Weibull.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

- Mann, N. R., Scheuer, E. M., and Fertig, K. W. (1973). *A new goodness-of-fit
  test for the two-parameter Weibull or extreme-value distribution with unknown
  parameters*. Communications in Statistics, 2(5), 383–400.
  [DOI](https://doi.org/10.1080/03610927308827085).
- Sürücü, B. (2008). *A power comparison and simulation study of goodness-of-fit
  tests*. Computers & Mathematics with Applications, 56(6), 1617–1625.
  [DOI](https://doi.org/10.1016/j.camwa.2008.03.010).
- Krit, M. (2014). *Goodness-of-fit tests for the Weibull distribution based on
  the Laplace transform*. Journal de la Société Française de Statistique,
  155(3), 135–151. [Article](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf).
