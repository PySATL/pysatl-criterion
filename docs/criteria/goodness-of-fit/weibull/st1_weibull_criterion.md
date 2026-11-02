# Best–Rayner–Thas skewness component for Weibull fit

`BestRaynerThasSkewnessWeibullGofStatistic` is also available as `ST1WeibullGofStatistic`.
The alias refers to the same class; its code remains `ST1_WEIBULL_GOODNESS_OF_FIT`.

## Definition

For a zero-location Weibull sample with unknown shape $k$ and scale $\eta$,
use $Y_i=-\log X_i$. Under the null, $Y$ is maximum Gumbel with
location $-\log\eta$ and scale $1/k$. Define

$$
m_j=\frac1n\sum_i(Y_i-\bar Y)^j,\qquad
 g_n=\frac{m_3}{m_2^{3/2}},\qquad b_n=\frac{m_4}{m_2^2}.
$$

All moments use divisor $n$; $b_n$ denotes Pearson kurtosis (not excess).
The implementation returns

$$
V_3^2=\frac{n(g_n-1.139547)^2}{20}.
$$

These are the rounded coefficients used by [EWGoF 2.2.2 source](https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WPP.test.R)
and reproduced in [Krit (2014), §3.2.5, (3.33)–(3.36)](https://theses.hal.science/tel-01126901/document).
The fourth component includes a correction for skewness. No maximum-likelihood
fit is used in either component.

## Names and compatibility

EWGoF's executable code defines **ST1 as skewness** and **ST2 as kurtosis**.
Its [help page](https://search.r-project.org/CRAN/refmans/EWGoF/html/WPP.test.html)
reverses these labels. The descriptive Python names remove this ambiguity;
the historical aliases follow the executable source.

The previous local delta-method discrepancies have been replaced. In particular,
ST1 no longer divides by the asymptotic variance of sample skewness, and ST2
no longer uses a covariance-based regression correction. Old critical values
must be regenerated despite the preserved short codes.

## Usage

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    BestRaynerThasSkewnessWeibullGofStatistic,
)

criterion = BestRaynerThasSkewnessWeibullGofStatistic(Weibull.DEFAULT.parse({}))
value = criterion.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
```

Both parameters must be unknown. Input must be one-dimensional, finite,
strictly positive and nonconstant, with at least two observations. Ties are
accepted. Invalid input raises `ValueError`; input arrays are not modified.

## Calibration

Large values reject. At fixed sample size, invariance under $X\mapsto aX^b$
for $a,b>0$ permits simulation with Weibull shape = scale = 1. Every replicate
must undergo the same negative-log transformation and moment standardization.
At $n=2$ the statistic is constant and cannot discriminate alternatives.
Rounded or censored observations require calibration for that observation scheme.

The return value is a statistic, not a p-value. The denominators are smooth
polynomial normalizations, not delta-method variances after estimating location
and scale. An unqualified $\chi^2_1$ approximation is not justified for these
rounded implementations. Use finite-sample Monte Carlo calibration.
Unversioned stored Weibull calibrations remain blocked.

## Scientific reference and verification scope

Best, D. J., Rayner, J. C. W., & Thas, O. (2007).
[Comparison of Five Tests of Fit for the Extreme Value Distribution](https://doi.org/10.1080/15598608.2007.10411826).
*Journal of Statistical Theory and Practice*, **1**(1), 89–99.
[Author-uploaded copy](https://www.researchgate.net/publication/266965257_Comparison_of_Five_Tests_of_Fit_for_the_Extreme_Value_Distribution).

The accessible author PDF contains printed pages 89–91 and 96–97 only;
pages 92–95 and the appendices needed for direct formula verification are missing.
Exact implementation coefficients were checked against EWGoF and Krit, not
against the missing pages. Agreement with every example in the original article
has therefore not been established.

See the [mathematical validation](mathematical-validation.md#st1-and-st2-follow-up-verification)
for reproducible numerical checks and the unresolved Port Pirie discrepancy.

For the 65 observations in the article's Table 1 (Port Pirie), treating them as
$Y$ and passing $X=\exp(-Y)$ gives $g_n=0.7110707539472332$ and
$V_3^2=0.5966736536522815$. Section 4 reports $V_3^2=0.71$.
The reported number is close to the sample skewness, but this alone does not
establish a typographical error. Replacing the statistic by sample skewness or
changing its denominator merely to reproduce 0.71 would change the test without
scientific justification. The discrepancy remains unresolved.
