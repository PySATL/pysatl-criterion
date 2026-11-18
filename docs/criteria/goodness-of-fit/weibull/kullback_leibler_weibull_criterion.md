# Kullback–Leibler information test for log-Weibull

`KullbackLeiblerWeibullGofStatistic` (`KL`) implements the MLE-based
Vasicek-entropy discrepancy for a Gumbel distribution, transferred to Weibull
through logarithms. Its functional matches EWGoF's `WEDF.test(type="KL")`
when the effective window and estimation method agree. Defaults differ.

## Hypothesis and API

The null is the ordinary zero-location Weibull family

$$
F(x;k,\eta)=1-\exp[-(x/\eta)^k],\qquad x>0,\quad k,\eta>0.
$$

Construct with `WeibullDistributionDescriptor.DEFAULT.parse({})`.
`execute_statistic(rvs, m=None, **kwargs)` requires a finite, strictly positive,
nonconstant one-dimensional complete sample, $n\ge3$. Both parameters are
estimated by maximum likelihood on every call. The sample is not modified;
estimates are not retained. Common-interface keywords are unused.

The entropy window must be an integer $1\le m<n/2$; booleans are invalid.
The default is $\min(\lfloor\sqrt n\rfloor,\lfloor(n-1)/2\rfloor)$.
Invalid samples or windows and unrepresentable fits raise `ValueError`.
Zero entropy-window gaps return positive infinity; ties whose window gaps
remain positive give finite results. Censoring is not supported.

## Gumbel transformation and statistic

$Y=\log X$ is **minimum Gumbel**, with location $\mu=\log\eta$ and scale
$\sigma=1/k$. For the ordered MLE-standardized logs

$$
z_i=\widehat k(\log X_{(i)}-\log\widehat\eta),
\qquad g(z)=\exp(z-e^z),
$$

define the endpoint-repeated Vasicek entropy estimator

$$
H_{m,n}=\frac1n\sum_{i=1}^n
\log\left[\frac{n}{2m}
\left(z_{\min(n,i+m)}-z_{\max(1,i-m)}\right)\right].
$$

The returned statistic is

$$
I_{m,n}=-H_{m,n}-\bar z+\frac1n\sum_{i=1}^n e^{z_i}.
$$

The signs follow directly from $\log g(z)=z-e^z$ and
$D(f\Vert g)=-H(f)-\mathbb E_f\log g$. This is a finite-sample entropy
estimate of divergence, not an exact divergence between two known densities;
it need not be nonnegative and is not clipped. There is no factor $n$, no
exponentiation and no additional bias correction in the returned value.
The MLE location score gives $n^{-1}\sum_i e^{z_i}=1$ up to numerical error.

Equivalently, applying the entropy estimator to unstandardized $Y$ gives
$H_{m,n}(z)=H_{m,n}(Y)-\log\widehat\sigma$ and
$I=-H_{m,n}(Y)-n^{-1}\sum_i\log f_Y(Y_i;\widehat\mu,\widehat\sigma)$.
For the maximum-Gumbel convention use $V=-\log X$; reflection reverses the
order but preserves the entropy-window sum and gives the same statistic
with the appropriately reflected density. Population KL divergence is
invariant under the invertible logarithmic change of variable. This does
not make a spacing entropy estimate on raw $X$ identical to one on $\log X$.

## Scientific sources and EWGoF correspondence

The Gumbel KL test is due to
[Pérez-Rodríguez, Vaquera-Huerta and Villaseñor-Alva (2009)](https://doi.org/10.1080/03610920802316658),
using the methodology of [Song (2002)](https://doi.org/10.1109/18.995548)
and a [Vasicek (1976)](https://doi.org/10.1111/j.2517-6161.1976.tb01566.x)
entropy estimate. The original 2009 full text was not checked directly;
exact computational correspondence was checked in the pinned EWGoF 2.2.2
[`WEDF.test` KL branch](https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/WEDF.test.R)
and [`MLEst`](https://github.com/cran/EWGoF/blob/51ef071b72a340b31807dacfa0e8f1d4a5a5eeb1/R/MLEst.R).
Although `MLEst` starts with `-log(x)`, it reflects again when standardizing,
so its returned values have the minimum-Gumbel orientation used here.

EWGoF also offers moment and least-squares fits; this class implements MLE only.
Our profile-score root solver differs from EWGoF's bounded optimization
(shape interval $[0.0001,50]$), so numerical outputs are not promised to be
bit-for-bit equal. More importantly, EWGoF's wrapper overwrites `paramKL`
using a piecewise sample-size rule. For example, at $n=20$ it chooses $m=3$,
while the local default is $m=4$. Its rule also contains uncovered boundary
sizes which fall through to $m=14$. This implementation retains its explicit
validated window rule and does not copy those boundary behaviours.

## Rejection and calibration

Large $I_{m,n}$ rejects (`RightAlternative`). Under $X\mapsto aX^b$, $a,b>0$,
logs undergo a positive affine transformation. MLE location and scale transform
accordingly, leaving standardized logs and the statistic unchanged for fixed $m$.
Thus calibration can simulate complete Weibull samples at shape = scale = 1.
Every replicate must refit both parameters and use the same window rule.

Use the upper $1-\alpha$ simulated quantile at the observed $n$. For explicit
`m`, supply matching external calibration: the generic resolver uses default
execution settings. Do not reuse EWGoF defaults or published critical values
without matching the window and estimator. No chi-square or unqualified normal
limit is asserted. Continuous-null simulation does not model rounding-induced
ties. See [mathematical verification](mathematical-validation.md#kullbackleibler-follow-up-verification).

## Example

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    KullbackLeiblerWeibullGofStatistic,
)

test = KullbackLeiblerWeibullGofStatistic(Weibull.DEFAULT.parse({}))
sample = [0.3, 0.6, 1.0, 1.4, 2.2]
print(test.execute_statistic(sample))
print(test.execute_statistic(sample, m=2))  # Requires calibration with m=2.
```

## References

- Pérez-Rodríguez, P., Vaquera-Huerta, H., and Villaseñor-Alva, J. A. (2009).
  *A Goodness-of-Fit Test for the Gumbel Distribution Based on Kullback–Leibler
  Information*. Communications in Statistics — Theory and Methods, 38(6),
  842–855. [DOI](https://doi.org/10.1080/03610920802316658).
- Vasicek, O. (1976). *A Test for Normality Based on Sample Entropy*.
  Journal of the Royal Statistical Society, Series B, 38(1), 54–59.
  [DOI](https://doi.org/10.1111/j.2517-6161.1976.tb01566.x).
- Song, K. S. (2002). *Goodness-of-fit tests based on Kullback-Leibler
  discrimination information*. IEEE Transactions on Information Theory,
  48(5), 1103–1117. [DOI](https://doi.org/10.1109/18.995548).
