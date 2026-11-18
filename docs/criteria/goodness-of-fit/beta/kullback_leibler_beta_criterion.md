# Vasicek–Kullback–Leibler Beta criterion

`KullbackLeiblerBetaGofStatistic`, code `KL_BETA_GOODNESS_OF_FIT`, estimates
KL divergence from a Beta density with both shapes fitted by maximum likelihood.
The support is fixed at $[0,1]$.

## Definition

For the ordered sample $X_{(1)},\ldots,X_{(n)}$, use repeated endpoints:
$X_{(j)}=X_{(1)}$ for $j<1$ and $X_{(j)}=X_{(n)}$ for $j>n$.
For an integer window $1\le m<n/2$, the Vasicek entropy estimate is

$$H_{m,n}=\frac1n\sum_{i=1}^n\log\left[
\frac{n}{2m}(X_{(i+m)}-X_{(i-m)})\right].$$

For the maximum-likelihood estimates $\hat a,\hat b$ the returned statistic is

$$I_{m,n}=-H_{m,n}+\log B(\hat a,\hat b)
-(\hat a-1)\frac1n\sum_i\log X_i
-(\hat b-1)\frac1n\sum_i\log(1-X_i).$$

This estimates $D(f\Vert f_{a,b})=-H(f)-E_f\log f_{a,b}(X)$.
There is no factor $n$ or exponentiation. Finite-sample estimates can be negative
and are **not clipped**. Log spacings are evaluated before adding the logarithm
of the scale factor, avoiding avoidable underflow. Sorting takes $O(n\log n)$
time and $O(n)$ memory, in addition to fitting.

## Parameters and calibration

Use `Beta.DEFAULT.parse({})`. Set the entropy window in the constructor with
`window=...`. With `window=None`, the local default is
$\min(\lfloor\sqrt n\rfloor,\lfloor(n-1)/2\rfloor)$; this is an explicit
implementation choice, not a claimed optimal rule from the article.
Booleans and noninteger windows are rejected. The bound $2m<n$ is checked
for each sample.

Require at least three finite, real observations strictly inside $(0,1)$ in a
nonconstant one-dimensional sample. Masked observations are rejected. Ties with
positive window gaps are accepted; zero gaps raise `ValueError`.
Input arrays are unchanged and fitted shapes are not retained.

Large values reject. Use `parametric_bootstrap_beta` to simulate the fitted Beta
and refit both shapes for each replicate while preserving the entropy-window
setting. Zero gaps are rejected before bootstrap calibration. It gives approximate calibration, not exact finite-sample size.
Do not reuse fixed-shape tables or silently optimize the window over the data.
Generic stored calibration is disabled for fitted Beta tests.

```python
from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    KullbackLeiblerBetaGofStatistic,
)
from pysatl_criterion.hypothesis_testing.beta_bootstrap import parametric_bootstrap_beta

sample = [0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57]
test = KullbackLeiblerBetaGofStatistic(Beta.DEFAULT.parse({}), window=2)
print(test.execute_statistic(sample))
# Small resample count for illustration; use more for actual inference.
result = parametric_bootstrap_beta(test, sample, n_resamples=99, random_state=42)
print(result.p_value)
```

## Scientific sources and verification limits

- H. Alizadeh Noughabi and N. R. Arghami (2013; online 2012), **General treatment
  of goodness-of-fit tests based on Kullback–Leibler information**, *Journal of
  Statistical Computation and Simulation*, 83(8), 1556–1569.
  [Scientific article](https://doi.org/10.1080/00949655.2012.667100).
- O. Vasicek (1976), **A test for normality based on sample entropy**, *Journal
  of the Royal Statistical Society, Series B*, 38(1), 54–59.
  [Entropy estimator](https://doi.org/10.1111/j.2517-6161.1976.tb01566.x).

The 2013 article describes the general Vasicek/KL method and its Beta application.
Its full text was unavailable during implementation. Exact agreement with its
Beta normalization, window selection and numerical tables is **not verified**.
The explicit raw-KL formula above follows directly from the Beta log density.
Tests compare it with SciPy's independent Vasicek estimator and an explicit
log-likelihood reference. See the [mathematical verification report](mathematical-validation.md).
