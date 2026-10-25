# Öztürk–Korukoğlu Weibull criterion

`OzturkKorukogluWeibullGofStatistic`, short code `OK`, computes the standardized
$OK_n^*$ statistic. It compares two linear estimates of the scale of log
observations. For Weibull shape $k$ and scale $\eta$, $\log X$ has the
smallest-extreme-value distribution with location $\log\eta$ and scale $1/k$.
The criterion eliminates these unknown parameters without an MLE fit.

## Formula

The definition follows [Öztürk and Korukoğlu (1988)](https://doi.org/10.1080/03610918808812730),
as reproduced in [Krit (2014), §3.2.2, equations (3.16)–(3.20)](https://theses.hal.science/tel-01126901/document).
Write $\ell_i=\log X_{(i)}$ and $y_i=\ell_i-\bar\ell$. For $i<n$ set

$$
w_i=\log\frac{n+1}{n-i+1},\qquad v_i=w_i(1+\log w_i)-1,
$$

and complete the weights with

$$
w_n=n-\sum_{i=1}^{n-1}w_i,\qquad
v_n=0.4228n-\sum_{i=1}^{n-1}v_i.
$$

The two scale estimates and the returned statistic are

$$
b=\frac1n\sum_{i=1}^n(0.6079v_i-0.2570w_i)y_i,
\qquad S=\frac{\sum_{i=1}^n(2i-n-1)y_i}{n(n-1)\log2},
$$

$$
T=\frac{b/S-1-0.13/\sqrt n+1.18/n}{0.49/\sqrt n-0.36/n}.
$$

### Centering adjustment

The source evaluates the numerator using uncentered $\ell_i$.
The rounded coefficients satisfy
$0.6079\cdot0.4228-0.2570=0.00002012$, so the implementation uses
$b=b_{\mathrm{raw}}-0.00002012\bar\ell$.
This explicitly removes the residual dependence on log location.
Under $X\mapsto cX^d$ with $c,d>0$, both $b$ and $S$ multiply by $d$;
therefore $T$ is invariant. This is the same adjustment used by
[Shapiro–Brain](shapiro_brain_weibull_criterion.md).

## Domain and calibration

Use a complete, finite, strictly positive, one-dimensional sample with $n\ge2$
and at least two distinct observations. Ties are accepted; invalid samples
raise `ValueError`. The input is not modified.

Since $\sum_i(2i-n-1)y_i=\sum_{i<j}(y_j-y_i)$, $S$ is positive for every
valid sample. Also $0.49/\sqrt n-0.36/n>0$ for $n\ge2$.
At $n=2$ the ratio is sample-independent and cannot discriminate distributions.

Reject for **small or large** values of $T$. Use lower and upper Monte Carlo
quantiles at the actual sample size, evaluating this centered implementation
on every replicate. Invariance permits simulation with shape = scale = 1.
The empirical normalization does not imply an exact standard-normal null law;
$T$ is signed and is not a squared discrepancy.

## Usage

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    OzturkKorukogluWeibullGofStatistic,
)

criterion = OzturkKorukogluWeibullGofStatistic(
    WeibullDistributionDescriptor.DEFAULT.parse({})
)
value = criterion.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
```

## References

- Öztürk, A. and Korukoğlu, S. (1988). *A new test for the extreme value
  distribution*. Communications in Statistics—Simulation and Computation,
  17(4), 1375–1393. [DOI](https://doi.org/10.1080/03610918808812730).
- Krit, M. (2014). *Goodness-of-fit tests in reliability: Weibull distribution
  and imperfect maintenance models*. PhD thesis, Université de Grenoble,
  §3.2.2, pp. 48–49. [Full text](https://theses.hal.science/tel-01126901/document).

See the [catalog](index.md) and [mathematical verification](mathematical-validation.md).
