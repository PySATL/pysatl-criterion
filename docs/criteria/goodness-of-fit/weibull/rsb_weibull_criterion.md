# Smith–Bain correlation criterion

`SmithBainWeibullGofStatistic`

## Constructor

```text
parameters : ParameterValues
    An empty distribution schema; all distribution parameters are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k.
Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
Logarithmic location and scale are eliminated within every call.

T=n*(1-r**2), where r=corr(log(x_(i)), log(-log(1-i/(n+1)))).

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
```

## Mathematical verification

Let $y_i=\log X_{(i)}$, $p_i=i/(n+1)$ and
$c_i=\log[-\log(1-p_i)]$. The implementation computes

$$
R^2=\frac{[\sum_i(y_i-\bar y)(c_i-\bar c)]^2}
{\sum_i(y_i-\bar y)^2\sum_i(c_i-\bar c)^2},\qquad T=n(1-R^2).
$$

This agrees with [Krit (2014), §3.2.1, equation (3.12), p. 47](https://theses.hal.science/tel-01126901/document),
which attributes the test to [Smith and Bain (1976)](https://doi.org/10.1080/03610927608827337).
The original article's full text was not checked directly. The scores are
quantiles at **mean uniform ranks**, not exact expected log order statistics:
in general $G^{-1}(E[U_{(i)}])\ne E[G^{-1}(U_{(i)})]$.
Changing the ranks changes the test and requires new calibration.

Our algebra gives $0\le T\le n$ by Cauchy–Schwarz. In a regression of $y_i$
on $c_i$ with an intercept, $T/n$ is the residual sum of squares divided by
the total centered sum of squares. Thus a poor fit gives **large** $T$.
Krit's accompanying phrase “too close to 1” is inconsistent with the stated
$Z_2=n(1-R^2)$; it must not be used as a rejection rule or a fixed threshold.

For $X=\eta E^{1/k}$ with $E\sim\operatorname{Exp}(1)$,
$\log X=\log\eta+k^{-1}\log E$. Correlation eliminates both parameters;
no MLE is performed. More generally, $X\mapsto aX^b$, $a,b>0$, preserves $T$.
The log-Weibull transformation and probability-plot construction are also
explained in [Krit (2014), §2, equations (3), (8)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf).

Calibrate the right tail for the actual sample size using independent complete
samples from $\operatorname{Exp}(1)$ (Weibull shape = scale = 1), recomputing
$T$ in every replicate. No chi-square or standard-normal null law is assumed.
At $n=2$, every nonconstant sample has $R^2=1$ and $T=0$ in exact arithmetic:
the API accepts it, but a nondegenerate test requires $n\ge3$.
Constant or nonpositive samples are rejected. Ties are accepted numerically;
rounded-data calibration must reproduce the observation/rounding mechanism.
Censoring is not supported by this implementation, despite the original article's title.

## Calling execute_statistic

```text
Compute the statistic described in the class Notes.

Parameters
----------
rvs : array_like
    One-dimensional finite strictly positive observations, n >= 2.
    Constant samples are invalid; ties are allowed unless stated below.
**kwargs : dict
    Unused common-interface keywords.

Returns
-------
statistic : float
    Finite discrepancy in [0, n], up to floating-point rounding.

Raises
------
ValueError
    Invalid sample, unsupported settings, or unrepresentable fit.

Notes
-----
The input is never changed and estimated parameters are not retained.
```

## Example

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import SmithBainWeibullGofStatistic

test = SmithBainWeibullGofStatistic(Distribution.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```


See [the mathematical verification](mathematical-validation.md#smithbain-follow-up-verification)
for source verification limits and numerical checks. Ordinary and exponentiated Weibull have separate distribution and criterion identities;
old critical values must not be reused after a formula or family change.

## References

- Smith, R. M. and Bain, L. J. (1976). *Correlation type goodness-of-fit statistics
  with censored sampling*. Communications in Statistics — Theory and Methods,
  5(2), 119–132. [DOI](https://doi.org/10.1080/03610927608827337).
- Krit, M. (2014). *Goodness-of-fit tests in reliability: Weibull distribution and
  imperfect maintenance models*. PhD thesis, Université de Grenoble, §3.2.1,
  equation (3.12), p. 47. [Full text](https://theses.hal.science/tel-01126901/document).
- Krit, M. (2014). *Goodness-of-fit tests for the Weibull distribution based on the
  Laplace transform*. Journal de la Société Française de Statistique, 155(3),
  135–151. [Full text](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf).
