# Evans–Johnson–Green correlation criterion

`EvansJohnsonGreenWeibullGofStatistic`

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

R2=corr(log(x_(i)), log(-log(1-p_i)))**2,
p_i=(i-0.3175)/(n+0.365). No fitted shape is needed: it cancels.

Reject for left tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
The returned statistic is R2, not R4 or n*(1-R2).
The former implementation returned R4; its stored critical values cannot
be reused unchanged.
```

## Mathematical verification

Write $y_i=\log X_{(i)}$, $p_i=(i-0.3175)/(n+0.365)$ and
$c_i=\log[-\log(1-p_i)]$, for every $i=1,\ldots,n$, including the endpoints.
The implementation computes

$$
R^2=\frac{[\sum_i(y_i-\bar y)(c_i-\bar c)]^2}
{\sum_i(y_i-\bar y)^2\sum_i(c_i-\bar c)^2}.
$$

The exact formula agrees with [Krit (2014), §3.2.1, (3.13), p. 47](https://theses.hal.science/tel-01126901/document),
which attributes it to Evans, Johnson and Green. Krit uses
$M_i=c_i/\widehat{k}$: the common positive factor cancels from $R^2$.
Also, $\sum_i(y_i-\bar y)M_i=\sum_i(y_i-\bar y)(M_i-\bar M)$.
Thus the implementation needs neither a fitted shape nor a fitted scale.

[Evans, Johnson and Green (1989), p. 3, equation (3)](https://www.govinfo.gov/content/pkg/GOVPUB-A13-PURL-gpo123139/pdf/GOVPUB-A13-PURL-gpo123139.pdf)
describe the log-scale correlation statistic for the two-parameter family,
distinct from the original-scale statistic in their equation (2).
The exact rank formula was checked through Krit; the original report's
formula images were not directly readable during this review.
[Filliben (1975)](https://doi.org/10.1080/00401706.1975.10489279)
is the historical probability-plot reference, not a source of Weibull critical values.
The ranks here are used at **all** indices; no special endpoint replacement is made.
They differ from Smith–Bain's $i/(n+1)$, so its calibration cannot be substituted.

Our algebra establishes $0\le R^2\le1$ by Cauchy–Schwarz. Both sequences are
ordered and nonconstant, so their covariance is positive; squaring does not
hide negative correlation. In a regression of $y_i$ on $c_i$ with an intercept,
$R^2=1-\mathrm{SSE}/\mathrm{SST}$. Poor linear fit lowers $R^2$, giving a
**left-tail** test. The former fourth-power implementation is not the stated
statistic; thresholds would need the corresponding transformation.

For $X=\eta E^{1/k}$, $E\sim\operatorname{Exp}(1)$,
$\log X=\log\eta+k^{-1}\log E$. Correlation removes the location and positive
scale, proving a parameter-free null law at fixed $n$ and invariance under
$X\mapsto aX^b$ for $a,b>0$. Simulate complete samples from
$\operatorname{Exp}(1)$ and recompute $R^2$ to calibrate its lower tail for the
actual sample size. No chi-square or normal approximation is asserted.

At $n=2$ every nonconstant sample has $R^2=1$ in exact arithmetic, so testing
requires $n\ge3$ even though the API accepts two observations. Ties are allowed
numerically, but rounding must be reproduced during calibration. Constant,
nonpositive and nonfinite observations are rejected. This implementation does
not support censoring or an unknown location parameter.

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
    Squared correlation in [0, 1], up to floating-point rounding.

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
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import EvansJohnsonGreenWeibullGofStatistic

test = EvansJohnsonGreenWeibullGofStatistic(Distribution.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```


See the [mathematical verification](mathematical-validation.md#evansjohnsongreen-follow-up-verification)
for source qualifications and numerical checks. Ordinary and exponentiated Weibull have separate distribution and criterion identities;
old critical values must not be reused after a formula or family change.

## References

- Evans, J. W., Johnson, R. A. and Green, D. W. (1989). *Two- and three-parameter
  Weibull goodness-of-fit tests*. USDA Forest Service, Research Paper FPL-RP-493,
  27 pp. [Full text](https://www.govinfo.gov/content/pkg/GOVPUB-A13-PURL-gpo123139/pdf/GOVPUB-A13-PURL-gpo123139.pdf).
- Krit, M. (2014). *Goodness-of-fit tests in reliability: Weibull distribution and
  imperfect maintenance models*. PhD thesis, Université de Grenoble, §3.2.1,
  (3.13). [Full text](https://theses.hal.science/tel-01126901/document).
- Filliben, J. J. (1975). *The probability plot correlation coefficient test for
  normality*. Technometrics, 17(1), 111–117.
  [DOI](https://doi.org/10.1080/00401706.1975.10489279).
