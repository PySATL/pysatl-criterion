# Zhang Z_C for a specified Beta distribution

[All Beta criteria](index.md) · [Mathematical verification](mathematical-validation.md)

Class: `ZhangCBetaGofStatistic`. Code: `ZHANG_ZC_BETA_GOODNESS_OF_FIT`.

## Hypothesis and formula

Both shape parameters `a > 0` and `b > 0` are finite and fixed independently of
the sample. Pass `Beta.DEFAULT.parse({"a": 2, "b": 5})`; omitted shapes are
unknown and are rejected by this class. The support is fixed at $[0,1]$.

For sorted observations, let $u_i=F_{a,b}(X_{(i)})$, $i=1,\ldots,n$,
$D^+=\max_i(i/n-u_i)$, $D^-=\max_i(u_i-(i-1)/n)$ and
$W^2=1/(12n)+\sum_i(u_i-(i-1/2)/n)^2$.

$$
Z_C=\sum_{i=1}^n\left[\log\frac{1-u_i}{u_i}-\log\frac{n-i+1/4}{i-3/4}\right]^2
$$


## Input, result and calibration

`execute_statistic(rvs, **kwargs)` accepts a nonempty, finite, one-dimensional
real sample in $[0,1]$. Ties and constants are allowed; masked observations and
complex values are rejected. It returns a scalar and leaves the sample unchanged.
Large values reject (`RightAlternative`). Extra execution keywords are ignored.

For a continuous fully specified null, the probability integral transform
makes this EDF statistic's null law independent of both shapes. Calibrate at the
same sample size. No parameter is fitted, and no p-value is returned.
AD, Liao–Shimokawa and Zhang use separate log-CDF and log-survival probabilities;
the other kernels use the CDF. Underflowed interior tails have a hypergeometric
logarithmic fallback; unrepresentable tail evaluation raises `ValueError`.
True endpoints give infinity for AD, Liao–Shimokawa and Zhang. Other EDF
statistics remain finite. The data are never clipped.

## Example

```python
from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import ZhangCBetaGofStatistic

statistic = ZhangCBetaGofStatistic(Beta.DEFAULT.parse({"a": 2, "b": 5}))
value = statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(value)
```

## Scientific source

[Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), equation (3.3).

The source identifies the shared statistical functional; applying it through
the specified Beta distribution does not constitute a new Beta-specific test.
